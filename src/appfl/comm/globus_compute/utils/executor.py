import time
from appfl.agent import ClientAgent
from appfl.comm.globus_compute.utils.client_utils import (
    load_global_model,
    send_local_model,
)


def get_sample_size_executor(
    client_agent_config=None,
    **kwargs,
):  
    task_sent_time = float(client_agent_config.start_time)
    total_task_sent_time = time.time() - task_sent_time
    print(f"serve to client time: {total_task_sent_time}")
    task_execution_start_time = time.time()
    client_agent = ClientAgent(client_agent_config=client_agent_config)
    sample_size = client_agent.get_sample_size()
    # return None, {"sample_size": client_agent.get_sample_size()}
    total_task_execution_time = time.time() - task_execution_start_time
    print(f"total get sample time: {total_task_execution_time}")
    return None, {
        "sample_size": sample_size, 
        "end_time": time.time(), 
        "server_to_client_time": total_task_sent_time,
        "total_model_download_time": "n/a",
        "total_execution_time": total_task_execution_time,
        "total_pre_val_time": "n/a",
        "total_forward_time": "n/a",
        "total_backward_time": "n/a",
        "total_val_time": "n/a",
    }


def data_readiness_report_executor(
    client_agent_config=None,
    **kwargs,
):
    client_agent = ClientAgent(client_agent_config=client_agent_config)
    return None, {
        "data_readiness": client_agent.generate_readiness_report(client_agent_config)
    }


def train_executor(
    client_agent_config=None,
    model=None,
    meta_data=None,
):
    # client_agent = ClientAgent(client_agent_config=client_agent_config)
    task_sent_time = float(client_agent_config.start_time)
    total_task_sent_time = time.time() - task_sent_time
    print(f"serve to client time: {total_task_sent_time}")
    
    donwload_model_start_time = time.time()
    if model is not None:
        # model = load_global_model(client_agent.client_agent_config, model)
        # client_agent.load_parameters(model)
        model = load_global_model(client_agent_config, model)
    total_model_download_time = time.time() - donwload_model_start_time
    print(f"total donwload model time: {total_model_download_time}")
    
    training_start_time = time.time()  
    client_agent = ClientAgent(client_agent_config=client_agent_config)
    client_agent.load_parameters(model)

    client_agent.train(**meta_data)
    local_model = client_agent.get_parameters()
    if isinstance(local_model, tuple):
        local_model, meta_data_local = local_model
    else:
        meta_data_local = {}
    local_model = send_local_model(
        client_agent.client_agent_config,
        local_model,
        meta_data["local_model_key"] if "local_model_key" in meta_data else None,
        meta_data["local_model_url"] if "local_model_url" in meta_data else None,
    )
    total_task_execution_time = time.time() - training_start_time

    meta_data_local['end_time'] = time.time()
    meta_data_local['server_to_client_time'] = total_task_sent_time
    meta_data_local['total_model_download_time'] = total_model_download_time
    meta_data_local['total_execution_time'] = total_task_execution_time
    meta_data_local['total_pre_val_time'] = client_agent.trainer.total_pre_val_time
    meta_data_local['total_forward_time'] = client_agent.trainer.total_forward_time
    meta_data_local['total_backward_time'] = client_agent.trainer.total_backward_time
    meta_data_local['total_val_time'] = client_agent.trainer.total_val_time
    return local_model, meta_data_local
