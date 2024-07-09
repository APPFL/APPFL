import argparse
from omegaconf import OmegaConf
from concurrent.futures import Future
from globus_compute_sdk import Client
from appfl.agent import ServerAgent
from appfl.communicator.globus_compute import GlobusComputeServerCommunicator

argparser = argparse.ArgumentParser()
argparser.add_argument("--config", type=str, default="configs/communication/server_gc_cnn.yaml")
argparser.add_argument("--num_clients", type=int, default=1)
args = argparser.parse_args()

# Load server agent configuration
server_agent_config = OmegaConf.load(args.config)
server_agent_config.server_configs.scheduler_kwargs.num_clients = args.num_clients

# Load client agents configuration
client_agent_config = OmegaConf.load("configs/communication/client_gc.yaml")
base_client_agent_config = client_agent_config['clients'][0].copy()
client_agent_config['clients'] = [base_client_agent_config for _ in range(args.num_clients)]
for i in range(args.num_clients):
    client_agent_config['clients'][i].client_id = f"client_{i}"
    client_agent_config['clients'][i].train_configs.logging_id = f'Client{i}'


# Create server agent
server_agent = ServerAgent(server_agent_config=server_agent_config)

# Create server communicator
gcc = Client()
server_communicator = GlobusComputeServerCommunicator(
    server_agent_config=server_agent.server_agent_config,
    client_agent_configs=client_agent_config['clients'],
    logger=server_agent.logger,
)

# Get sample size from clients
server_communicator.send_task_to_all_clients(task_name="get_sample_size")
sample_size_ret = server_communicator.recv_result_from_all_clients()[1]
for client_id, sample_size in sample_size_ret.items():
    server_agent.set_sample_size(client_id, sample_size['sample_size'])

# Train the model
server_communicator.send_task_to_all_clients(
    task_name="train",
    model=server_agent.get_parameters(globus_compute_run=True),
    need_model_response=True,
)

model_futures = {}
while not server_agent.training_finished():
    client_id, client_model, client_metadata = server_communicator.recv_result_from_one_client()
    global_model = server_agent.global_update(
        client_id,
        client_model,
        **client_metadata,
    )
    if isinstance(global_model, Future):
        model_futures[client_id] = global_model
    else:
        if isinstance(global_model, tuple):
            global_model, metadata = global_model
        else:
            metadata = {}
        if not server_agent.training_finished():
            server_communicator.send_task_to_one_client(
                client_id,
                task_name="train",
                model=global_model,
                metadata=metadata,
                need_model_response=True,
            )
    # Deal with the model futures
    del_keys = []
    for client_id in model_futures:
        if model_futures[client_id].done():
            global_model = model_futures[client_id].result()
            if isinstance(global_model, tuple):
                global_model, metadata = global_model
            else:
                metadata = {}
            if not server_agent.training_finished():
                server_communicator.send_task_to_one_client(
                    client_id,
                    task_name="train",
                    model=global_model,
                    metadata=metadata,
                    need_model_response=True,
                )
            del_keys.append(client_id)
    for key in del_keys:
        model_futures.pop(key)

server_communicator.cancel_all_tasks()
server_communicator.shutdown_all_clients()