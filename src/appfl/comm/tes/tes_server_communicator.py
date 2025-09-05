import json
import uuid
import time
import requests
from typing import Dict, List, Optional, Union, OrderedDict, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, Future, as_completed

from appfl.comm.base import BaseServerCommunicator
from appfl.comm.utils.config import ClientTask
from appfl.comm.utils.s3_storage import CloudStorage, LargeObjectWrapper
from appfl.config import ClientAgentConfig, ServerAgentConfig
from appfl.logger import ServerAgentFileLogger


class TESServerCommunicator(BaseServerCommunicator):
    """
    GA4GH Task Execution Service (TES) server communicator for APPFL.
    
    This communicator enables APPFL to submit federated learning tasks to
    GA4GH TES-compliant compute infrastructures, following the same patterns
    as Globus Compute and Ray communicators.
    """
    
    def __init__(
        self,
        server_agent_config: ServerAgentConfig,
        client_agent_configs: List[ClientAgentConfig],
        logger: Optional[ServerAgentFileLogger] = None,
        **kwargs
    ):
        self.comm_type = "tes"
        super().__init__(server_agent_config, client_agent_configs, logger, **kwargs)
        
        # Load TES-specific configuration
        self._load_tes_configs()
        
        # Initialize TES client endpoints mapping
        self.client_endpoints: Dict[str, Dict] = {}
        _client_id_check_set = set()
        
        for client_config in client_agent_configs:
            # Get client ID
            client_id = str(
                client_config.client_id
                if hasattr(client_config, "client_id")
                else client_config.train_configs.logging_id
                if hasattr(client_config.train_configs, "logging_id")
                else f"tes_client_{len(self.client_endpoints)}"
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
            client_config.comm_type = self.comm_type
            
            # Store client endpoint info
            self.client_endpoints[client_id] = {
                "client_id": client_id,
                "client_config": client_config,
                "resource_requirements": self._get_client_resources(client_config),
            }
        
        # Thread pool for managing TES task submissions
        self.executor = ThreadPoolExecutor(max_workers=len(client_agent_configs))
        self.logger.info(f"TES Server Communicator initialized with endpoint: {self.tes_endpoint}")
        self.logger.info(f"Managing {len(self.client_endpoints)} client endpoints")

    def _load_tes_configs(self):
        """Load TES-specific configuration from server config."""
        tes_configs = {}
        if (hasattr(self.server_agent_config.server_configs, "comm_configs") and 
            hasattr(self.server_agent_config.server_configs.comm_configs, "tes_configs")):
            tes_configs = self.server_agent_config.server_configs.comm_configs.tes_configs
        
        self.tes_endpoint = tes_configs.get("tes_endpoint", "http://localhost:8000")
        self.auth_token = tes_configs.get("auth_token", None)
        self.docker_image = tes_configs.get("docker_image", "appfl/client:latest")
        
        # Default resource requirements
        default_resources = {
            "cpu_cores": 1,
            "ram_gb": 2.0,
            "disk_gb": 10.0,
            "preemptible": False
        }
        self.default_resource_requirements = {
            **default_resources, 
            **tes_configs.get("resource_requirements", {})
        }

    def _get_client_resources(self, client_config: ClientAgentConfig) -> Dict:
        """Get resource requirements for a specific client."""
        # Check if client has specific resource requirements
        if (hasattr(client_config, "resource_configs") and 
            hasattr(client_config.resource_configs, "tes_resources")):
            return {**self.default_resource_requirements, **client_config.resource_configs.tes_resources}
        return self.default_resource_requirements

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
            "python", "-m", "appfl.run_tes_client",
            "--task-name", task_name,
            "--client-id", client_id,
        ]
        
        # Handle model input
        if model is not None:
            model_path = f"/tmp/model_{task_id}.pkl"
            if isinstance(model, bytes):
                model_content = model
            else:
                import pickle
                model_content = pickle.dumps(model)
            
            inputs.append({
                "name": "model_data",
                "description": "Serialized model parameters",
                "path": model_path,
                "type": "FILE",
                "content": model_content.hex()  # TES expects hex-encoded content
            })
            command_args.extend(["--model-path", model_path])
        
        # Handle metadata input
        if metadata is not None:
            metadata_path = f"/tmp/metadata_{task_id}.json"
            inputs.append({
                "name": "task_metadata",
                "description": "Task metadata",
                "path": metadata_path,
                "type": "FILE",
                "content": json.dumps(metadata)
            })
            command_args.extend(["--metadata-path", metadata_path])
        
        # Prepare task outputs
        output_model_path = f"/tmp/output_model_{task_id}.pkl"
        output_logs_path = f"/tmp/training_logs_{task_id}.json"
        
        outputs = [
            {
                "name": "trained_model",
                "path": output_model_path,
                "type": "FILE"
            },
            {
                "name": "training_logs",
                "path": output_logs_path,
                "type": "FILE"
            }
        ]
        
        command_args.extend([
            "--output-path", output_model_path,
            "--logs-path", output_logs_path
        ])
        
        # Create TES task
        tes_task = {
            "name": f"appfl-{task_name}-{client_id}-{task_id[:8]}",
            "description": f"APPFL federated learning task: {task_name} for client {client_id}",
            "inputs": inputs,
            "outputs": outputs,
            "executors": [{
                "image": self.docker_image,
                "command": command_args,
                "workdir": "/tmp"
            }],
            "resources": {
                "cpu_cores": client_info["resource_requirements"]["cpu_cores"],
                "ram_gb": client_info["resource_requirements"]["ram_gb"],
                "disk_gb": client_info["resource_requirements"]["disk_gb"],
                "preemptible": client_info["resource_requirements"]["preemptible"]
            },
            "tags": {
                "appfl_experiment_id": self.experiment_id,
                "appfl_client_id": client_id,
                "appfl_task_name": task_name,
                "appfl_task_id": task_id
            }
        }
        
        return tes_task

    def _submit_tes_task(self, tes_task: Dict) -> str:
        """Submit a TES task and return the task ID."""
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        response = requests.post(
            f"{self.tes_endpoint}/ga4gh/tes/v1/tasks",
            json=tes_task,
            headers=headers
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"TES task submission failed: {response.status_code} - {response.text}")
        
        result = response.json()
        return result["id"]

    def _get_tes_task_status(self, tes_task_id: str) -> Dict:
        """Get TES task status."""
        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        response = requests.get(
            f"{self.tes_endpoint}/ga4gh/tes/v1/tasks/{tes_task_id}",
            headers=headers
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Failed to get TES task status: {response.status_code} - {response.text}")
        
        return response.json()

    def _wait_for_task_completion(self, tes_task_id: str, timeout: int = 3600) -> Tuple[Any, Dict]:
        """Wait for TES task completion and return results."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            task_info = self._get_tes_task_status(tes_task_id)
            state = task_info.get("state", "UNKNOWN")
            
            if state == "COMPLETE":
                # Task completed successfully, extract results
                return self._extract_task_results(task_info)
            elif state in ["SYSTEM_ERROR", "EXECUTOR_ERROR", "CANCELED"]:
                logs = task_info.get("logs", [])
                error_msg = f"TES task {tes_task_id} failed with state: {state}"
                if logs:
                    error_msg += f"\nLogs: {logs}"
                raise RuntimeError(error_msg)
            
            time.sleep(5)  # Poll every 5 seconds
        
        raise TimeoutError(f"TES task {tes_task_id} timed out after {timeout} seconds")

    def _extract_task_results(self, task_info: Dict) -> Tuple[Any, Dict]:
        """Extract model and metadata from completed TES task."""
        outputs = task_info.get("outputs", [])
        
        model_result = None
        metadata_result = {}
        
        for output in outputs:
            if output["name"] == "trained_model":
                # Download and deserialize model
                if "content" in output:
                    import pickle
                    model_bytes = bytes.fromhex(output["content"])
                    model_result = pickle.loads(model_bytes)
            elif output["name"] == "training_logs":
                # Parse training logs
                if "content" in output:
                    metadata_result = json.loads(output["content"])
        
        return model_result, metadata_result

    def send_task_to_all_clients(
        self,
        task_name: str,
        *,
        model: Optional[Union[Dict, OrderedDict, bytes]] = None,
        metadata: Union[Dict, List[Dict]] = {},
        need_model_response: bool = False,
    ):
        """
        Send a specific task to all clients via TES.
        """
        # Handle S3 model storage if enabled
        if self.use_s3bucket and model is not None:
            model_wrapper = LargeObjectWrapper(
                data=model,
                name=str(uuid.uuid4()) + "_server_state",
            )
            model = CloudStorage.upload_object(model_wrapper, register_for_clean=True)
        
        for i, client_id in enumerate(self.client_endpoints):
            client_metadata = metadata[i] if isinstance(metadata, list) else metadata
            
            # Add S3 upload URL for model response if needed
            if need_model_response and self.use_s3bucket:
                local_model_key = f"{str(uuid.uuid4())}_client_state_{client_id}"
                local_model_url = CloudStorage.presign_upload_object(local_model_key)
                client_metadata["local_model_key"] = local_model_key
                client_metadata["local_model_url"] = local_model_url
            
            # Create and submit TES task
            tes_task = self._create_tes_task(client_id, task_name, model, client_metadata)
            tes_task_id = self._submit_tes_task(tes_task)
            
            # Create a future for this task
            task_future = self.executor.submit(self._wait_for_task_completion, tes_task_id)
            
            # Register the task
            self._register_task(tes_task_id, task_future, client_id, task_name)
            self.logger.info(f"TES task '{task_name}' (ID: {tes_task_id}) assigned to {client_id}")

    def send_task_to_one_client(
        self,
        client_id: str,
        task_name: str,
        *,
        model: Optional[Union[Dict, OrderedDict, bytes]] = None,
        metadata: Optional[Dict] = {},
        need_model_response: bool = False,
    ):
        """
        Send a specific task to one specific client via TES.
        """
        if client_id not in self.client_endpoints:
            raise ValueError(f"Client ID {client_id} not found in configured endpoints")
        
        # Handle S3 model storage if enabled
        if self.use_s3bucket and model is not None:
            model_wrapper = LargeObjectWrapper(
                data=model,
                name=str(uuid.uuid4()) + "_server_state",
            )
            model = CloudStorage.upload_object(model_wrapper, register_for_clean=True)
        
        # Add S3 upload URL for model response if needed
        if need_model_response and self.use_s3bucket:
            local_model_key = f"{str(uuid.uuid4())}_client_state_{client_id}"
            local_model_url = CloudStorage.presign_upload_object(local_model_key)
            metadata["local_model_key"] = local_model_key
            metadata["local_model_url"] = local_model_url
        
        # Create and submit TES task
        tes_task = self._create_tes_task(client_id, task_name, model, metadata)
        tes_task_id = self._submit_tes_task(tes_task)
        
        # Create a future for this task
        task_future = self.executor.submit(self._wait_for_task_completion, tes_task_id)
        
        # Register the task
        self._register_task(tes_task_id, task_future, client_id, task_name)
        self.logger.info(f"TES task '{task_name}' (ID: {tes_task_id}) assigned to {client_id}")

    def recv_result_from_all_clients(self) -> Tuple[Dict, Dict]:
        """
        Receive task results from all clients that have running tasks.
        """
        client_results, client_metadata = {}, {}
        
        # Wait for all tasks to complete
        for task_future in as_completed(self.executing_task_futs.keys()):
            task_id = self.executing_task_futs[task_future]
            task_info = self.executing_tasks[task_id]
            client_id = task_info.client_id
            
            try:
                # Get result from the completed task
                model_result, metadata_result = task_future.result()
                
                # Handle S3 model download if needed
                if self.use_s3bucket and metadata_result.get("local_model_key"):
                    model_result = CloudStorage.download_object(metadata_result["local_model_key"])
                
                client_results[client_id] = model_result
                client_metadata[client_id] = metadata_result
                
                # Log completion
                elapsed = time.time() - task_info.start_time
                self.logger.info(
                    f"Task '{task_info.task_name}' from {client_id} completed in {elapsed:.2f}s"
                )
                
            except Exception as e:
                self.logger.error(f"Task from {client_id} failed: {e}")
                client_results[client_id] = None
                client_metadata[client_id] = {"error": str(e)}
            
            # Clean up
            del self.executing_task_futs[task_future]
            del self.executing_tasks[task_id]
        
        return client_results, client_metadata

    def recv_result_from_one_client(self) -> Tuple[str, Any, Dict]:
        """
        Receive task results from the first client that finishes the task.
        """
        if not self.executing_task_futs:
            raise RuntimeError("No tasks are currently executing")
        
        # Wait for the first task to complete
        completed_future = next(as_completed(self.executing_task_futs.keys()))
        task_id = self.executing_task_futs[completed_future]
        task_info = self.executing_tasks[task_id]
        client_id = task_info.client_id
        
        try:
            # Get result from the completed task
            model_result, metadata_result = completed_future.result()
            
            # Handle S3 model download if needed
            if self.use_s3bucket and metadata_result.get("local_model_key"):
                model_result = CloudStorage.download_object(metadata_result["local_model_key"])
            
            # Log completion
            elapsed = time.time() - task_info.start_time
            self.logger.info(
                f"Task '{task_info.task_name}' from {client_id} completed in {elapsed:.2f}s"
            )
            
        except Exception as e:
            self.logger.error(f"Task from {client_id} failed: {e}")
            model_result = None
            metadata_result = {"error": str(e)}
        
        # Clean up
        del self.executing_task_futs[completed_future]
        del self.executing_tasks[task_id]
        
        return client_id, model_result, metadata_result

    def shutdown_all_clients(self):
        """Cancel all running TES tasks and shutdown the thread pool executor."""
        # Cancel all running tasks
        self.cancel_all_tasks()
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True)
        self.logger.info("TES Server Communicator shutdown complete")

    def cancel_all_tasks(self):
        """Cancel all on-the-fly TES tasks."""
        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        for task_id in list(self.executing_tasks.keys()):
            try:
                # Cancel TES task
                response = requests.post(
                    f"{self.tes_endpoint}/ga4gh/tes/v1/tasks/{task_id}:cancel",
                    headers=headers
                )
                if response.status_code == 200:
                    self.logger.info(f"TES task {task_id} cancelled successfully")
                else:
                    self.logger.warning(f"Failed to cancel TES task {task_id}: {response.status_code}")
            except Exception as e:
                self.logger.error(f"Error cancelling TES task {task_id}: {e}")
        
        # Clear all executing tasks
        self.executing_tasks.clear()
        self.executing_task_futs.clear()