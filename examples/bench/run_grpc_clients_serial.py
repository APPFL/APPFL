"""
Run multiple gRPC clients to benchmark the server.
"""

import time
import argparse
from omegaconf import OmegaConf
from appfl.agent import ClientAgent
from appfl.communicator.grpc import GRPCClientCommunicator
from concurrent.futures import ThreadPoolExecutor

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--num_clients",
    type=int,
    default=1,
    help="Number of clients."
)
argparser.add_argument(
    "--client_config", 
    type=str, 
    default="configs/communication/client_grpc.yaml",
    help="Path to the configuration file."
)
argparser.add_argument(
    '--server_uri',
    type=str,
    required=False,
)
args = argparser.parse_args()

num_clients = args.num_clients

grpc_executor = ThreadPoolExecutor(max_workers=num_clients)

client_agent_configs = [OmegaConf.load(args.client_config) for _ in range(num_clients)]
if args.server_uri:
    for client_agent_config in client_agent_configs:
        client_agent_config.comm_configs.grpc_configs.server_uri = args.server_uri

for i in range(num_clients):
    client_agent_configs[i].train_configs.logging_id = f'Client{i+1}'
    client_agent_configs[i].data_configs.dataset_kwargs.num_clients = num_clients
    client_agent_configs[i].data_configs.dataset_kwargs.client_id = i
    client_agent_configs[i].data_configs.dataset_kwargs.visualization = False
    
client_agents = [
    ClientAgent(client_agent_config=client_agent_configs[i]) 
    for i in range(num_clients)
]

client_agent_communicators = [
    GRPCClientCommunicator(
        client_id = client_agents[i].get_id(),
        **client_agent_configs[i].comm_configs.grpc_configs,
    )
    for i in range(num_clients)
]

# Load additional client configurations from the server
client_config_from_server = client_agent_communicators[0].get_configuration()
for client_agent in client_agents:
    client_agent.load_config(client_config_from_server)
    

# Load initial global model from the server
init_model = client_agent_communicators[0].get_global_model(serial_run=True)
for client_agent in client_agents:
    client_agent.load_parameters(init_model)

# [Optional] Set number of local data to the server
sample_sizes = [client_agents[i].get_sample_size() for i in range(num_clients)]
sample_size_futures = [
    grpc_executor.submit(
        client_agent_communicators[i].invoke_custom_action,
        action='set_sample_size',
        sample_size=sample_sizes[i]
    )
    for i in range(num_clients)
]
sample_size_futures = [f.result() for f in sample_size_futures]

start_time = time.time()

client_agents[0].train()
local_model = client_agents[0].get_parameters()
while True:
    global_model_futures = [
        grpc_executor.submit(
            client_agent_communicators[i].update_global_model,
            local_model
        )
        for i in range(num_clients)
    ]
    for i in range(num_clients):
        if i == 0:
            new_global_model, metadata = global_model_futures[i].result()
        else:
            res = global_model_futures[i].result()
            del res
    if metadata['status'] == 'DONE':
        break
    
for i in range(num_clients):
    client_agent_communicators[i].invoke_custom_action(action='close_connection')
    client_agents[i].clean_up()

end_time = time.time()
print(f"Total time taken: {end_time - start_time:.2f} seconds")