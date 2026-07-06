import os
import pathlib
from proxystore.store import Store
from proxystore.proxy import Proxy, extract
from appfl.misc.utils import get_proxystore_connector
from appfl.comm.utils.s3_storage import CloudStorage
from typing import Optional, Tuple, Any, Dict


class ModelTransferHelper:
    """
    Stateless helper for server-side model-transfer storage setup and result
    parsing, shared by the server-side communication components (the
    ``BaseServerCommunicator`` and the transport-specific ``ServerCommBackend``
    implementations). It currently covers AWS S3 and ProxyStore.

    These are plain functions grouped under a class: callers invoke them and assign
    the returned values to their own attributes, rather than inheriting behavior via
    a mixin. This keeps the communicator and backend class hierarchies free of an
    extra base class they don't otherwise need.
    """

    @staticmethod
    def init_s3(
        server_agent_config,
        logger,
        comm_type: str,
        experiment_id: str,
    ) -> bool:
        """
        Configure AWS S3 model transfer from the server config (with backward
        compatibility for the deprecated ``globus_compute_configs`` key) and
        initialize ``CloudStorage`` if enabled.

        :return `use_s3bucket`: Whether S3 model transfer is enabled.
        """
        use_s3bucket = False
        s3_bucket = None
        if hasattr(server_agent_config.server_configs, "comm_configs") and hasattr(
            server_agent_config.server_configs.comm_configs, "s3_configs"
        ):
            use_s3bucket = server_agent_config.server_configs.comm_configs.s3_configs.get(
                "enable_s3", False
            )
            s3_bucket = server_agent_config.server_configs.comm_configs.s3_configs.get(
                "s3_bucket", None
            )
            use_s3bucket = use_s3bucket and s3_bucket is not None
        # backward compatibility for globus compute
        if (
            hasattr(server_agent_config.server_configs, "comm_configs")
            and hasattr(
                server_agent_config.server_configs.comm_configs,
                "globus_compute_configs",
            )
            and hasattr(
                server_agent_config.server_configs.comm_configs.globus_compute_configs,
                "s3_bucket",
            )
        ):
            logger.warning(
                "[Deprecation] Use of globus_compute_configs in server configs is deprecated. Moving forward use s3_configs key to configure AWS S3 you can find new examples here https://github.com/APPFL/APPFL/blob/main/examples/resources/config_gc/"
            )
            s3_bucket = server_agent_config.server_configs.comm_configs.globus_compute_configs.get(
                "s3_bucket", None
            )
            use_s3bucket = s3_bucket is not None
            # copy globus_compute_configs to s3_configs
            server_agent_config.server_configs.comm_configs.s3_configs = (
                server_agent_config.server_configs.comm_configs.globus_compute_configs
            )
            server_agent_config.server_configs.comm_configs.s3_configs["enable_s3"] = (
                use_s3bucket
            )

        if use_s3bucket:
            logger.info(f"Using S3 bucket {s3_bucket} for model transfer.")
            s3_creds_file = (
                server_agent_config.server_configs.comm_configs.s3_configs.get(
                    "s3_creds_file", None
                )
            )
            s3_temp_dir_default = str(
                pathlib.Path.home() / ".appfl" / comm_type / "server" / experiment_id
            )
            s3_temp_dir = (
                server_agent_config.server_configs.comm_configs.s3_configs.get(
                    "s3_temp_dir", s3_temp_dir_default
                )
            )
            if not os.path.exists(s3_temp_dir):
                pathlib.Path(s3_temp_dir).mkdir(parents=True, exist_ok=True)
            CloudStorage.init(s3_bucket, s3_creds_file, s3_temp_dir, logger)

        return use_s3bucket

    @staticmethod
    def init_proxystore(
        server_agent_config,
        logger,
    ) -> Tuple[bool, Optional[Store]]:
        """
        Create a ProxyStore for storing and sending model parameters from the server
        to the clients, if enabled in the server config.

        :return `(use_proxystore, proxystore)`.
        """
        proxystore = None
        use_proxystore = False
        if (
            hasattr(server_agent_config.server_configs, "comm_configs")
            and hasattr(
                server_agent_config.server_configs.comm_configs, "proxystore_configs"
            )
            and server_agent_config.server_configs.comm_configs.proxystore_configs.get(
                "enable_proxystore", False
            )
        ):
            use_proxystore = True
            proxystore = Store(
                name="server-proxystore",
                connector=get_proxystore_connector(
                    server_agent_config.server_configs.comm_configs.proxystore_configs.connector_type,
                    server_agent_config.server_configs.comm_configs.proxystore_configs.connector_configs,
                ),
            )
            logger.info(
                f"Server using proxystore for model transfer with store: {server_agent_config.server_configs.comm_configs.proxystore_configs.connector_type}."
            )
        return use_proxystore, proxystore

    @staticmethod
    def parse_result(result, use_s3bucket: bool) -> Tuple[Any, Dict]:
        """
        Parse the returned results from the client.
        The results can be composed of two parts:
        - Model parameters (can be model, gradients, compressed model, etc.)
        - Metadata (may contain additional information such as logs, etc.)

        :param `result`: The result returned from the client.
        :param `use_s3bucket`: Whether S3 model transfer is enabled.
        :return `model`: The model parameters returned from the client.
        :return `metadata`: The metadata returned from the client.
        """
        if isinstance(result, tuple):
            model, metadata = result
        else:
            model, metadata = result, {}
        # Download model from S3 bucket or ProxyStore if necessary
        if isinstance(model, Proxy):
            model = extract(model)
        if use_s3bucket:
            if CloudStorage.is_cloud_storage_object(model):
                model = CloudStorage.download_object(
                    model, delete_cloud=True, delete_local=True
                )
        return model, metadata
