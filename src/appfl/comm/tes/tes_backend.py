import os
import json
import uuid
import time
import yaml
import boto3
import requests
from omegaconf import OmegaConf
from botocore.exceptions import ClientError
from appfl.logger import ServerAgentFileLogger
from appfl.comm.base import ServerCommBackend
from appfl.config import ClientAgentConfig, ServerAgentConfig
from appfl.comm.utils.s3_storage import CloudStorage, LargeObjectWrapper
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union, OrderedDict, Tuple, Any


class TESBackend(ServerCommBackend):
    """
    GA4GH Task Execution Service (TES) server communication backend for APPFL.

    This backend enables APPFL to submit federated learning tasks to GA4GH
    TES-compliant compute infrastructures. It is driven by the generic
    :class:`~appfl.comm.base.ServerDrivenCommunicator`, which owns task
    bookkeeping and the unified receive loop.

    :param `server_agent_config`: The server agent configuration.
    :param `client_agent_configs`: The subset of client configurations served via TES.
    :param `experiment_id`: Shared experiment id assigned by the driver.
    :param [Optional] `logger`: Logger object shared with the driver.
    """

    def __init__(
        self,
        server_agent_config: ServerAgentConfig,
        client_agent_configs: List[ClientAgentConfig],
        experiment_id: str,
        logger: Optional[ServerAgentFileLogger] = None,
        **kwargs,
    ):
        super().__init__(
            comm_type="tes",
            server_agent_config=server_agent_config,
            client_agent_configs=client_agent_configs,
            experiment_id=experiment_id,
            logger=logger,
            **kwargs,
        )
        # The base S3 (CloudStorage) model-transfer path is temporarily disabled for
        # TES; use `comm_configs.tes_configs.file_storage: s3` instead. To re-enable,
        # restore the ModelTransferHelper.init_s3(...) call below.
        #   self.use_s3bucket = ModelTransferHelper.init_s3(
        #       server_agent_config, self.logger, self.comm_type, self.experiment_id
        #   )
        self.use_s3bucket = False
        self._warn_if_base_s3_enabled(server_agent_config)
        # Note: TES does not use ProxyStore for model transfer (it never has).
        self._warn_if_proxystore_enabled(server_agent_config)

        # Initialize TES client endpoints mapping
        self.client_endpoints: Dict[str, Dict] = {}
        _client_id_check_set = set()

        for client_config in client_agent_configs:
            # Get client ID
            client_id = str(
                client_config.client_id
                if hasattr(client_config, "client_id")
                else client_config.train_configs.logging_id
                if (
                    hasattr(client_config, "train_configs")
                    and hasattr(client_config.train_configs, "logging_id")
                )
                else f"default_tes_client_id_{len(self.client_endpoints)}"
            )

            assert client_id not in _client_id_check_set, (
                f"Client ID {client_id} is not unique for this client configuration.\n{client_config}"
            )
            _client_id_check_set.add(client_id)

            # Read client dataset source if provided
            if hasattr(client_config.data_configs, "dataset_path"):
                with open(client_config.data_configs.dataset_path) as file:
                    client_config.data_configs.dataset_source = file.read()
                del client_config.data_configs.dataset_path

            client_config.experiment_id = self.experiment_id
            client_config.comm_configs.comm_type = self.comm_type

            # Build the config shipped to the TES container and strip cloud transfer
            # mechanisms it must not use (see _disable_cloud_transfer_for_client).
            merged_client_config = OmegaConf.merge(
                server_agent_config.client_configs, client_config
            )
            self._disable_cloud_transfer_for_client(merged_client_config)

            # Store client endpoint info with per-client TES settings
            self.client_endpoints[client_id] = {
                "client_id": client_id,
                "client_config": merged_client_config,
                "resource_requirements": self._get_client_resources(
                    client_config, client_id
                ),
                "tes_endpoint": self._get_client_tes_endpoint(client_config, client_id),
                "auth_token": self._get_client_auth_token(client_config, client_id),
                "docker_image": self._get_client_docker_image(client_config, client_id),
            }

        # Thread pool for managing TES task submissions
        self.executor = ThreadPoolExecutor(max_workers=len(client_agent_configs))

        # Set up file storage configuration
        self._setup_file_storage(server_agent_config)

        # Set up Funnel workspace directory for local file transfers
        if self.file_storage_type == "local":
            self.funnel_workspace = self.file_storage_kwargs.get(
                "workspace_dir", "/tmp/funnel-workspace"
            )
            os.makedirs(self.funnel_workspace, exist_ok=True)
            self.logger.info(
                f"Using local file storage with workspace: {self.funnel_workspace}"
            )
            self.logger.info(
                f"Make sure to start Funnel with: --LocalStorage.AllowedDirs {self.funnel_workspace}"
            )
        else:
            self.funnel_workspace = (
                "/tmp/funnel-workspace"  # Still needed for temp operations
            )
            os.makedirs(self.funnel_workspace, exist_ok=True)

        # Log initialization with multi-endpoint support
        self.logger.info(
            f"Backend manages {len(self.client_endpoints)} client endpoint(s)."
        )
        for client_id, endpoint_info in self.client_endpoints.items():
            self.logger.info(
                f"Client '{client_id}' -> {endpoint_info['tes_endpoint']}"
            )

    def _disable_cloud_transfer_for_client(self, client_config):
        """
        Disable APPFL CloudStorage (S3) and ProxyStore in the config shipped to a
        TES client.

        TES transfers models through its own file storage (local Funnel workspace
        or presigned S3 URLs), not APPFL CloudStorage/ProxyStore. In a hybrid
        federation the shared `client_configs.comm_configs` may enable these (e.g.,
        for Globus Compute clients). If that leaks into a TES client's config, the
        in-container client code (`send_local_model`/`load_global_model`) would try
        a CloudStorage/ProxyStore transfer it cannot perform. Force them off so the
        TES client relies solely on the TES file-storage path.
        """
        comm_configs = getattr(client_config, "comm_configs", None)
        if comm_configs is None:
            return
        if hasattr(comm_configs, "s3_configs") and comm_configs.s3_configs.get(
            "enable_s3", False
        ):
            comm_configs.s3_configs.enable_s3 = False
        if hasattr(
            comm_configs, "proxystore_configs"
        ) and comm_configs.proxystore_configs.get("enable_proxystore", False):
            comm_configs.proxystore_configs.enable_proxystore = False

    def _warn_if_base_s3_enabled(self, server_agent_config: ServerAgentConfig):
        """Warn that the base S3 (CloudStorage) path is configured but disabled for TES."""
        comm_configs = getattr(server_agent_config.server_configs, "comm_configs", None)
        if (
            comm_configs is not None
            and hasattr(comm_configs, "s3_configs")
            and comm_configs.s3_configs.get("enable_s3", False)
        ):
            self.logger.warning(
                "`comm_configs.s3_configs.enable_s3` is set but the base S3 "
                "(CloudStorage) path is currently disabled for TES; it will be "
                "ignored. Use `comm_configs.tes_configs.file_storage: s3` instead."
            )

    def _warn_if_proxystore_enabled(self, server_agent_config: ServerAgentConfig):
        """Warn that ProxyStore is configured but unsupported by the TES transport."""
        comm_configs = getattr(server_agent_config.server_configs, "comm_configs", None)
        if (
            comm_configs is not None
            and hasattr(comm_configs, "proxystore_configs")
            and comm_configs.proxystore_configs.get("enable_proxystore", False)
        ):
            self.logger.warning(
                "ProxyStore is enabled in the server config but is not supported by "
                "the TES communicator; it will be ignored. Models are transferred via "
                "TES file storage (or AWS S3 if enabled)."
            )

    @property
    def client_ids(self) -> List[str]:
        return list(self.client_endpoints.keys())

    def _setup_file_storage(self, server_agent_config: ServerAgentConfig):
        """Set up file storage configuration from server config."""
        # Get file storage configuration from server config
        if hasattr(server_agent_config.server_configs, "comm_configs") and hasattr(
            server_agent_config.server_configs.comm_configs, "tes_configs"
        ):
            tes_configs = server_agent_config.server_configs.comm_configs.tes_configs
        else:
            tes_configs = {}
        self.file_storage_type = tes_configs.get("file_storage", "local")
        self.file_storage_kwargs = tes_configs.get("file_storage_kwargs", {})

        # Guard against enabling two overlapping S3 mechanisms at once: the base
        # `use_s3bucket` (CloudStorage, via s3_configs) and the TES-native
        # `file_storage: s3` (via tes_configs.file_storage_kwargs). Combining them
        # is unsupported and breaks model transfer. When both are set, prefer the
        # TES-native `file_storage: s3` and disable `use_s3bucket`.
        if self.use_s3bucket and self.file_storage_type == "s3":
            self.logger.warning(
                "Both `comm_configs.s3_configs.enable_s3` (CloudStorage) and "
                "`comm_configs.tes_configs.file_storage: s3` are enabled for TES. "
                "These are conflicting S3 mechanisms; `use_s3bucket` will be ignored "
                "in favor of the TES-native `file_storage: s3`."
            )
            self.use_s3bucket = False

        # Set up S3 client if using S3 storage
        if self.file_storage_type == "s3":
            self._setup_s3_client()
            self.logger.info(
                f"Using S3 file storage with bucket: {self.file_storage_kwargs.get('s3_bucket')}"
            )

        self.uploaded_files = set()  # Track uploaded files for cleanup

    def _setup_s3_client(self):
        """Initialize S3 client for S3 file storage."""
        s3_bucket = self.file_storage_kwargs.get("s3_bucket")
        if not s3_bucket:
            raise ValueError(
                "s3_bucket is required to be provided in server_configs.comm_configs.tes_configs.file_storage_kwargs for S3 file storage"
            )

        s3_region = self.file_storage_kwargs.get("s3_region", "us-east-1")
        aws_access_key_id = self.file_storage_kwargs.get("aws_access_key_id")
        aws_secret_access_key = self.file_storage_kwargs.get("aws_secret_access_key")
        self.presigned_url_expiry = self.file_storage_kwargs.get(
            "presigned_url_expiry", 3600
        )

        # Initialize S3 client
        s3_kwargs = {"region_name": s3_region}
        if aws_access_key_id and aws_secret_access_key:
            s3_kwargs.update(
                {
                    "aws_access_key_id": aws_access_key_id,
                    "aws_secret_access_key": aws_secret_access_key,
                }
            )

        self.s3_client = boto3.client("s3", **s3_kwargs)
        self.s3_bucket = s3_bucket

        # Verify bucket exists
        try:
            self.s3_client.head_bucket(Bucket=s3_bucket)
        except ClientError as e:
            error_code = int(e.response["Error"]["Code"])
            if error_code == 404:
                raise ValueError(f"S3 bucket '{s3_bucket}' does not exist")
            else:
                raise ValueError(f"Cannot access S3 bucket '{s3_bucket}': {e}")

    def _get_client_tes_endpoint(
        self, client_config: ClientAgentConfig, client_id: str
    ) -> str:
        """Get TES endpoint for a specific client."""
        if (
            hasattr(client_config, "comm_configs")
            and hasattr(client_config.comm_configs, "tes_configs")
            and hasattr(client_config.comm_configs.tes_configs, "tes_endpoint")
        ):
            return client_config.comm_configs.tes_configs.tes_endpoint
        raise ValueError(
            f"Client {client_id} missing required comm_configs.tes_configs.tes_endpoint"
        )

    def _get_client_auth_token(
        self, client_config: ClientAgentConfig, client_id: str
    ) -> Optional[str]:
        """Get auth token for a specific client's TES endpoint."""
        if (
            hasattr(client_config, "comm_configs")
            and hasattr(client_config.comm_configs, "tes_configs")
            and hasattr(client_config.comm_configs.tes_configs, "auth_token")
        ):
            auth_token = client_config.comm_configs.tes_configs.auth_token
            return auth_token if auth_token else None
        return None

    def _get_client_docker_image(
        self, client_config: ClientAgentConfig, client_id: str
    ) -> str:
        """Get Docker image for a specific client."""
        if (
            hasattr(client_config, "comm_configs")
            and hasattr(client_config.comm_configs, "tes_configs")
            and hasattr(client_config.comm_configs.tes_configs, "docker_image")
        ):
            return client_config.comm_configs.tes_configs.docker_image
        raise ValueError(
            f"Client {client_id} missing required comm_configs.tes_configs.docker_image"
        )

    def _get_client_resources(
        self, client_config: ClientAgentConfig, client_id: str
    ) -> Dict:
        """Get resource requirements for a specific client."""
        if (
            hasattr(client_config, "comm_configs")
            and hasattr(client_config.comm_configs, "tes_configs")
            and hasattr(client_config.comm_configs.tes_configs, "resource_requirements")
        ):
            return client_config.comm_configs.tes_configs.resource_requirements
        raise ValueError(
            f"Client {client_id} missing required comm_configs.tes_configs.resource_requirements"
        )

    def _get_client_volumes(self, client_config: ClientAgentConfig) -> List[Dict]:
        """Get volume mount configuration for a specific client's data access."""
        volumes = []

        # Check for client-specific volume configuration
        if hasattr(client_config, "data_configs") and hasattr(
            client_config.data_configs, "volume_mounts"
        ):
            for volume in client_config.data_configs.volume_mounts:
                volumes.append(
                    {
                        "name": volume.get("name", f"data_volume_{len(volumes)}"),
                        "source": volume.get("host_path", volume.get("source")),
                        "target": volume.get(
                            "container_path", volume.get("target", "/data")
                        ),
                        "readonly": volume.get("read_only", True),
                    }
                )

        # Check for simple data path mapping
        if hasattr(client_config, "data_configs") and hasattr(
            client_config.data_configs, "local_data_path"
        ):
            volumes.append(
                {
                    "name": "client_data",
                    "source": client_config.data_configs.local_data_path,
                    "target": "/data",
                    "readonly": True,
                }
            )

        return volumes

    def _get_client_environment(
        self, client_config: ClientAgentConfig
    ) -> Dict[str, str]:
        """Get environment variables for client's data access."""
        env_vars = {}

        # Check for client-specific environment variables
        if hasattr(client_config, "data_configs") and hasattr(
            client_config.data_configs, "environment"
        ):
            env_vars.update(client_config.data_configs.environment)

        # Add data path if volume mount is configured
        volumes = self._get_client_volumes(client_config)
        if volumes:
            env_vars["DATA_PATH"] = volumes[0]["target"]

        # Add client ID for data partitioning
        env_vars["CLIENT_ID"] = getattr(client_config, "client_id", "unknown")

        return env_vars

    def _upload_model(self, model, storage_key: str) -> Dict:
        """Upload model using configured file storage."""
        if self.file_storage_type == "local":
            return self._upload_model_local(model, storage_key)
        elif self.file_storage_type == "s3":
            return self._upload_model_s3(model, storage_key)
        else:
            raise ValueError(f"Unsupported file storage type: {self.file_storage_type}")

    def _upload_model_local(self, model, storage_key: str) -> Dict:
        """Upload model to local workspace directory."""
        import torch

        workspace_file_path = os.path.join(self.funnel_workspace, storage_key)

        if isinstance(model, bytes):
            with open(workspace_file_path, "wb") as f:
                f.write(model)
        else:
            torch.save(model, workspace_file_path)

        self.uploaded_files.add(workspace_file_path)
        return {
            "type": "file",
            "url": f"file://{workspace_file_path}",
            "storage_key": storage_key,
        }

    def _upload_model_s3(self, model, storage_key: str) -> Dict:
        """Upload model to S3 and return presigned URL."""
        import torch
        import tempfile
        from botocore.exceptions import ClientError

        s3_key = f"appfl-tes/{storage_key}"

        try:
            # Save model to temporary file for S3 upload
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                if isinstance(model, bytes):
                    temp_file.write(model)
                else:
                    torch.save(model, temp_file)
                temp_file_path = temp_file.name

            # Upload file to S3
            self.s3_client.upload_file(temp_file_path, self.s3_bucket, s3_key)
            self.uploaded_files.add(s3_key)

            # Clean up temporary file
            os.remove(temp_file_path)

            # Generate presigned download URL for client
            download_url = self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.s3_bucket, "Key": s3_key},
                ExpiresIn=self.presigned_url_expiry,
            )

            return {
                "type": "s3",
                "url": download_url,
                "storage_key": storage_key,
                "s3_bucket": self.s3_bucket,
                "s3_key": s3_key,
            }

        except ClientError as e:
            raise RuntimeError(f"Failed to upload model to S3: {e}")

    def _download_s3_result(self, s3_key: str, local_path: str):
        """Download result file from S3 to local path."""
        from botocore.exceptions import ClientError

        try:
            self.s3_client.download_file(self.s3_bucket, s3_key, local_path)
        except ClientError as e:
            raise RuntimeError(f"Failed to download {s3_key} from S3: {e}")

    def _cleanup_storage(self):
        """Clean up uploaded files."""
        if self.file_storage_type == "local":
            for file_path in self.uploaded_files:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup file {file_path}: {e}")
        elif self.file_storage_type == "s3":
            for s3_key in self.uploaded_files:
                try:
                    self.s3_client.delete_object(Bucket=self.s3_bucket, Key=s3_key)
                    self.logger.info(
                        f"Cleaned up S3 object: s3://{self.s3_bucket}/{s3_key}"
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup S3 object {s3_key}: {e}")
        self.uploaded_files.clear()

    def _create_tes_task(
        self,
        client_id: str,
        task_name: str,
        model: Optional[Union[Dict, OrderedDict, bytes]] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """Create a GA4GH TES task specification."""
        task_id = str(uuid.uuid4())
        client_info = self.client_endpoints[client_id]

        # Prepare task inputs
        inputs = []
        command_args = [
            "python",
            "-c",
            "from appfl.comm.tes.tes_client_communicator import tes_client_entry_point; tes_client_entry_point()",
            "--task-name",
            task_name,
            "--client-id",
            client_id,
        ]

        # Add client configuration
        client_config = client_info["client_config"]
        client_config_path = f"/app/configs/client_config_{task_id}.yaml"
        config_dict = (
            OmegaConf.to_container(client_config, resolve=True)
            if hasattr(client_config, "_content")
            else client_config
        )

        inputs.append(
            {
                "name": "client_config",
                "description": "Client configuration file",
                "path": client_config_path,
                "type": "FILE",
                "content": yaml.safe_dump(config_dict),
            }
        )
        command_args.extend(["--config-path", client_config_path])

        # Handle model input using configured file storage
        if model is not None:
            model_path = f"/app/configs/model_{task_id}.pkl"
            storage_info = self._upload_model(model, f"input_model_{task_id}.pkl")

            inputs.append(
                {
                    "name": "model_data",
                    "description": "Serialized model parameters",
                    "path": model_path,
                    "type": "FILE",
                    "url": storage_info["url"],
                }
            )
            command_args.extend(["--model-path", model_path])

        # Use /tmp inside container (writable), configure outputs based on storage type
        container_model_path = f"/tmp/output_model_{task_id}.pkl"
        container_logs_path = f"/tmp/training_logs_{task_id}.json"

        outputs = []
        s3_upload_info = None

        # Configure outputs based on file storage type
        if self.file_storage_type == "s3":
            # S3 storage - client will upload directly using presigned URLs
            # No TES output URLs needed, client handles S3 upload

            # Store S3 info for client to use for direct upload
            if not hasattr(self, "_upload_info"):
                self._upload_info = {}
            model_s3_key = f"appfl-tes/output_model_{task_id}.pkl"
            logs_s3_key = f"appfl-tes/output_logs_{task_id}.json"

            # Generate presigned upload URLs for client
            from botocore.exceptions import ClientError

            try:
                model_upload_url = self.s3_client.generate_presigned_url(
                    "put_object",
                    Params={"Bucket": self.s3_bucket, "Key": model_s3_key},
                    ExpiresIn=self.presigned_url_expiry,
                )

                logs_upload_url = self.s3_client.generate_presigned_url(
                    "put_object",
                    Params={"Bucket": self.s3_bucket, "Key": logs_s3_key},
                    ExpiresIn=self.presigned_url_expiry,
                )

                s3_upload_info = {
                    "model_upload_url": model_upload_url,
                    "logs_upload_url": logs_upload_url,
                    "model_s3_key": model_s3_key,
                    "logs_s3_key": logs_s3_key,
                }

                self._upload_info[task_id] = s3_upload_info

                # Track S3 keys for cleanup
                self.uploaded_files.add(model_s3_key)
                self.uploaded_files.add(logs_s3_key)

            except ClientError as e:
                raise RuntimeError(f"Failed to generate presigned upload URLs: {e}")
        else:
            # Local storage - use file URLs to funnel workspace
            outputs.extend(
                [
                    {
                        "name": "trained_model",
                        "path": container_model_path,
                        "url": f"file://{self.funnel_workspace}/model_{task_id}.pkl",
                        "type": "FILE",
                    },
                    {
                        "name": "training_logs",
                        "path": container_logs_path,
                        "url": f"file://{self.funnel_workspace}/results_{task_id}.json",
                        "type": "FILE",
                    },
                ]
            )

        # Handle metadata input
        client_metadata = metadata.copy() if metadata else {}

        # Add S3 upload info to metadata if using S3 storage
        if s3_upload_info:
            client_metadata["s3_upload_info"] = s3_upload_info

        if client_metadata:
            metadata_path = f"/app/configs/metadata_{task_id}.json"
            inputs.append(
                {
                    "name": "task_metadata",
                    "description": "Task metadata",
                    "path": metadata_path,
                    "type": "FILE",
                    "content": json.dumps(client_metadata),
                }
            )
            command_args.extend(["--metadata-path", metadata_path])

        command_args.extend(
            ["--output-path", container_model_path, "--logs-path", container_logs_path]
        )

        # Get client-specific volumes and environment
        volumes = self._get_client_volumes(client_config)
        environment = self._get_client_environment(client_config)

        # Create executor with volumes and environment
        executor = {
            "image": client_info["docker_image"],
            "command": command_args,
            "workdir": "/tmp",
        }

        # Add environment variables if specified
        if environment:
            executor["env"] = environment

        # Note: Volume mounts are not supported in standard TES/Funnel
        # For now, skip volume mounting until we find the correct TES approach
        if volumes:
            self.logger.warning(
                "Volume mounting requested but not supported by current TES implementation"
            )
            self.logger.warning(f"Requested volumes: {volumes}")
            # TODO: Implement proper TES volume mounting or alternative data access

        # Create TES task
        tes_task = {
            "name": f"appfl-{task_name}-{client_id}-{task_id[:8]}",
            "description": f"APPFL federated learning task: {task_name} for client {client_id}",
            "inputs": inputs,
            "outputs": outputs,
            "executors": [executor],
            "resources": {
                "cpu_cores": client_info["resource_requirements"]["cpu_cores"],
                "ram_gb": client_info["resource_requirements"]["ram_gb"],
                "disk_gb": client_info["resource_requirements"]["disk_gb"],
                "preemptible": client_info["resource_requirements"]["preemptible"],
            },
            "tags": {
                "appfl_experiment_id": self.experiment_id,
                "appfl_client_id": client_id,
                "appfl_task_name": task_name,
                "appfl_task_id": task_id,
            },
        }

        return tes_task

    def _submit_tes_task(self, tes_task: Dict, client_id: str) -> str:
        """Submit a TES task to the appropriate endpoint for the client."""
        client_info = self.client_endpoints[client_id]
        tes_endpoint = client_info["tes_endpoint"]
        auth_token = client_info["auth_token"]

        headers = {"Content-Type": "application/json"}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        response = requests.post(
            f"{tes_endpoint}/ga4gh/tes/v1/tasks", json=tes_task, headers=headers
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"TES task submission failed for {client_id} at {tes_endpoint}: {response.status_code} - {response.text}"
            )

        result = response.json()
        return result["id"]

    def _get_tes_task_status(self, tes_task_id: str, client_id: str) -> Dict:
        """Get TES task status from the appropriate endpoint."""
        client_info = self.client_endpoints[client_id]
        tes_endpoint = client_info["tes_endpoint"]
        auth_token = client_info["auth_token"]

        headers = {}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        # Add view=FULL to get complete task info including logs
        response = requests.get(
            f"{tes_endpoint}/ga4gh/tes/v1/tasks/{tes_task_id}?view=FULL",
            headers=headers,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to get TES task status for {client_id}: {response.status_code} - {response.text}"
            )

        return response.json()

    def _wait_for_task_completion(
        self, tes_task_id: str, client_id: str, timeout: int = 3600
    ) -> Tuple[Any, Dict]:
        """Wait for TES task completion and return results."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            task_info = self._get_tes_task_status(tes_task_id, client_id)
            state = task_info.get("state", "UNKNOWN")

            if state == "COMPLETE":
                # Task completed successfully, extract results
                return self._extract_task_results(task_info)
            elif state in ["SYSTEM_ERROR", "EXECUTOR_ERROR", "CANCELED"]:
                logs = task_info.get("logs", [])
                error_msg = (
                    f"TES task {tes_task_id} for {client_id} failed with state: {state}"
                )
                if logs:
                    error_msg += f"\nLogs: {logs}"
                raise RuntimeError(error_msg)

            time.sleep(2)  # Poll every 2 seconds

        raise TimeoutError(
            f"TES task {tes_task_id} for {client_id} timed out after {timeout} seconds"
        )

    def _extract_task_results(self, task_info: Dict) -> Tuple[Any, Dict]:
        """Extract real results from TES task outputs."""
        model_result = None
        metadata_result = {}
        task_id = task_info.get("tags", {}).get("appfl_task_id")

        # Handle S3 storage results
        if (
            self.file_storage_type == "s3"
            and hasattr(self, "_upload_info")
            and task_id in self._upload_info
        ):
            try:
                # Download model result from S3
                upload_info = self._upload_info[task_id]
                model_s3_key = upload_info["model_s3_key"]
                logs_s3_key = upload_info["logs_s3_key"]

                import tempfile

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pkl"
                ) as temp_file:
                    temp_model_path = temp_file.name

                self._download_s3_result(model_s3_key, temp_model_path)

                # Load model from downloaded file
                import torch

                model_result = torch.load(temp_model_path)
                os.remove(temp_model_path)  # Cleanup temp file

                # Download logs result from S3
                with tempfile.NamedTemporaryFile(
                    mode="w+", delete=False, suffix=".json"
                ) as temp_file:
                    temp_logs_path = temp_file.name

                self._download_s3_result(logs_s3_key, temp_logs_path)

                # Load logs from downloaded file
                with open(temp_logs_path) as f:
                    metadata_result = json.load(f)
                os.remove(temp_logs_path)  # Cleanup temp file

                # Clean up upload info
                del self._upload_info[task_id]

                return model_result, metadata_result

            except Exception as e:
                self.logger.error(f"Failed to download S3 results: {e}")
                # Fall back to checking if files were somehow put in local outputs

        # Handle local storage results (original logic)
        outputs = task_info.get("outputs", [])

        if not outputs:
            self.logger.warning(
                "No outputs found for the TES task; Funnel may not have processed "
                "the output files yet."
            )
            return model_result, metadata_result

        # Try to read from outputs
        for output in outputs:
            output_name = output.get("name", "")
            file_url = output.get("url", "")

            if output_name == "training_logs":
                # Check if Funnel put content directly in output
                if "content" in output:
                    try:
                        metadata_result = json.loads(output["content"])
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to parse training logs from output content: {e}"
                        )

                # Also check file path if available
                if file_url.startswith("file://"):
                    file_path = file_url[7:]  # Remove "file://" prefix
                    try:
                        if os.path.exists(file_path):
                            with open(file_path) as f:
                                metadata_result = json.load(f)
                            # Clean up the file after reading
                            os.remove(file_path)
                        else:
                            self.logger.warning(
                                f"Training logs file not found: {file_path}"
                            )
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to read training logs from {file_path}: {e}"
                        )

            elif output_name == "trained_model":
                # Handle model file cleanup
                if file_url.startswith("file://"):
                    file_path = file_url[7:]
                    try:
                        if os.path.exists(file_path):
                            with open(file_path, "rb") as f:
                                import pickle

                                model_result = pickle.load(f)
                            # Clean up the model file after reading
                            os.remove(file_path)
                        else:
                            self.logger.warning(
                                f"Trained model file not found: {file_path}"
                            )
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to load trained model from {file_path}: {e}"
                        )

        if not metadata_result:
            self.logger.warning("No results found in any TES task output.")

        return model_result, metadata_result

    def prepare_model(
        self, model: Optional[Union[Dict, OrderedDict, bytes]]
    ) -> Any:
        """Upload the shared model once to the base S3 store if enabled."""
        if model is None:
            return None
        if self.use_s3bucket:
            model_wrapper = LargeObjectWrapper(
                data=model,
                name=str(uuid.uuid4()) + "_server_state",
            )
            return CloudStorage.upload_object(model_wrapper, register_for_clean=True)
        return model

    def submit_task(
        self,
        client_id: str,
        task_name: str,
        prepared_model: Any = None,
        metadata: Optional[Dict] = None,
        need_model_response: bool = False,
    ) -> Tuple[str, Any]:
        if client_id not in self.client_endpoints:
            raise ValueError(f"Client ID {client_id} not found in configured endpoints")

        client_metadata = dict(metadata) if metadata else {}

        # Add S3 upload URL for model response if needed
        if need_model_response and self.use_s3bucket:
            local_model_key = f"{str(uuid.uuid4())}_client_state_{client_id}"
            local_model_url = CloudStorage.presign_upload_object(local_model_key)
            client_metadata["local_model_key"] = local_model_key
            client_metadata["local_model_url"] = local_model_url

        # Create and submit TES task to client's specific endpoint
        tes_task = self._create_tes_task(
            client_id, task_name, prepared_model, client_metadata
        )
        tes_task_id = self._submit_tes_task(tes_task, client_id)

        # Create a future for this task
        task_future = self.executor.submit(
            self._wait_for_task_completion, tes_task_id, client_id
        )

        # Log with client's specific TES endpoint
        client_endpoint = self.client_endpoints[client_id]["tes_endpoint"]
        # self.logger.info(
        #     f"TES task '{task_name}' (ID: {tes_task_id}) submitted to {client_id} at {client_endpoint}"
        # )
        return tes_task_id, task_future

    def process_result(self, future: Any) -> Tuple[Any, Dict]:
        model_result, metadata_result = future.result()

        # Handle base S3 model download if needed
        if self.use_s3bucket and metadata_result.get("local_model_key"):
            model_result = CloudStorage.download_object(
                metadata_result["local_model_key"]
            )
        return model_result, metadata_result

    def cancel_task(self, task_id: str, future: Any, client_id: str):
        """Cancel a single on-the-fly TES task."""
        try:
            client_endpoint_info = self.client_endpoints[client_id]
            tes_endpoint = client_endpoint_info["tes_endpoint"]
            auth_token = client_endpoint_info["auth_token"]

            headers = {}
            if auth_token:
                headers["Authorization"] = f"Bearer {auth_token}"

            response = requests.post(
                f"{tes_endpoint}/ga4gh/tes/v1/tasks/{task_id}:cancel",
                headers=headers,
            )
            if response.status_code == 200:
                self.logger.info(
                    f"TES task {task_id} for {client_id} cancelled successfully at {tes_endpoint}"
                )
            else:
                self.logger.warning(
                    f"Failed to cancel TES task {task_id} for {client_id}: {response.status_code}"
                )
        except Exception as e:
            self.logger.error(f"Error cancelling TES task {task_id}: {e}")

    def shutdown(self):
        """Clean up uploaded files and shutdown the thread pool executor."""
        self._cleanup_storage()
        self.executor.shutdown(wait=True)
        self.logger.info("Backend shutdown complete.")
