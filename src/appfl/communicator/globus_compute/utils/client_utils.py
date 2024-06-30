import pathlib
from appfl.config import ClientAgentConfig
from typing import Union, Dict, OrderedDict, Any
from .s3_storage import CloudStorage, LargeObjectWrapper

def load_global_model(
    client_agent_config: ClientAgentConfig,
    global_model: Any
):
    s3_tmp_dir = str(
        pathlib.Path.home() / ".appfl" / "globus_compute" / client_agent_config.endpoint_id / client_agent_config.experiment_id
    )
    if not pathlib.Path(s3_tmp_dir).exists():
        pathlib.Path(s3_tmp_dir).mkdir(parents=True, exist_ok=True)
    if CloudStorage.is_cloud_storage_object(global_model):
        CloudStorage.init(s3_tmp_dir=s3_tmp_dir)
        global_model = CloudStorage.download_object(global_model)
    return global_model

def send_local_model(
    client_agent_config: ClientAgentConfig,
    local_model:  Union[Dict, OrderedDict, bytes],
    local_model_key: str,
    local_model_url: str,
):
    if (
        hasattr(client_agent_config.comm_configs, "globus_compute_configs") and
        client_agent_config.comm_configs.globus_compute_configs.get("s3_bucket", None) is not None
    ):
        s3_tmp_dir = str(
            pathlib.Path.home() / ".appfl" / "globus_compute" / client_agent_config.endpoint_id / client_agent_config.experiment_id
        )
        if not pathlib.Path(s3_tmp_dir).exists():
            pathlib.Path(s3_tmp_dir).mkdir(parents=True, exist_ok=True)
        local_model_wrapper = LargeObjectWrapper(local_model, local_model_key)
        if not local_model_wrapper.can_send_directly:
            CloudStorage.init(s3_tmp_dir=s3_tmp_dir)
            local_model = CloudStorage.upload_object(
                local_model_wrapper, 
                object_url=local_model_url, 
                ext='pt' if not isinstance(local_model, bytes) else 'pkl'
            )
    return local_model
