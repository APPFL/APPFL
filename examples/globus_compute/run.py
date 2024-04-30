from omegaconf import OmegaConf
from concurrent.futures import Future
from globus_compute_sdk import Client
from appfl.agent import APPFLServerAgent
from appfl.comm.globus_compute import GlobusComputeServerCommunicator

# Load server and client agents configurations
server_agent_config = OmegaConf.load("config_gc/server_fedavg.yaml")
client_agent_configs = OmegaConf.load("config_gc/clients.yaml")

# Create server agent
server_agent = APPFLServerAgent(server_agent_config=server_agent_config)

# Create server communicator
gcc = Client()
server_communicator = GlobusComputeServerCommunicator(
    server_agent_config=server_agent.server_agent_config,
    client_agent_configs=client_agent_configs['clients'],
    logger=server_agent.logger,
)

# Get sample size from clients
server_communicator.send_task_to_all_clients(task_name="get_sample_size")
sample_size_ret = server_communicator.recv_result_from_all_clients()[1]
for client_endpoint_id, sample_size in sample_size_ret.items():
    server_agent.set_sample_size(client_endpoint_id, sample_size['sample_size'])

# Train the model
server_communicator.send_task_to_all_clients(
    task_name="train",
    model=server_agent.get_parameters(globus_compute_run=True),
    need_model_response=True,
)

model_futures = {}
while not server_agent.training_finished():
    client_endpoint_id, client_model, client_metadata = server_communicator.recv_result_from_one_client()
    global_model = server_agent.global_update(
        client_endpoint_id,
        client_model,
        **client_metadata,
    )
    if isinstance(global_model, Future):
        model_futures[client_endpoint_id] = global_model
    else:
        if isinstance(global_model, tuple):
            global_model, metadata = global_model
        else:
            metadata = {}
        server_communicator.send_task_to_one_client(
            client_endpoint_id,
            task_name="train",
            model=global_model,
            metadata=metadata,
            need_model_response=True,
        )
    # Deal with the model futures
    del_keys = []
    for client_endpoint_id in model_futures:
        if model_futures[client_endpoint_id].done():
            global_model = model_futures[client_endpoint_id].result()
            if isinstance(global_model, tuple):
                global_model, metadata = global_model
            else:
                metadata = {}
            server_communicator.send_task_to_one_client(
                client_endpoint_id,
                task_name="train",
                model=global_model,
                metadata=metadata,
                need_model_response=True,
            )
            del_keys.append(client_endpoint_id)
    for key in del_keys:
        model_futures.pop(key)

server_communicator.cancel_all_tasks()
server_communicator.shutdown_all_clients()