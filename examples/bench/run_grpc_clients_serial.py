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

client_agent_config = OmegaConf.load(args.client_config)
if args.server_uri:
    client_agent_config.comm_configs.grpc_configs.server_uri = args.server_uri
    
client_agent = ClientAgent(client_agent_config=client_agent_config)
client_agent_communicator = GRPCClientCommunicator(
    client_id = 0,
    **client_agent_config.comm_configs.grpc_configs,
)

client_config_from_server = client_agent_communicator.get_configuration()
client_agent.load_config(client_config_from_server)

init_model = client_agent_communicator.get_global_model(serial_run=True)
client_agent.load_parameters(init_model)

start_time = time.time()

client_agent.train()
local_model = client_agent.get_parameters()
while True:
    global_model_futures = []
    for i in range(num_clients):
        print(f"Client {i} is updating the global model.")
        global_model_futures.append(
            grpc_executor.submit(
                client_agent_communicator.update_global_model,
                local_model,
                _client_id=i
            )
        )
    for i in range(num_clients):
        if i == 0:
            new_global_model, metadata = global_model_futures[i].result()
        else:
            res = global_model_futures[i].result()
            del res
    if metadata['status'] == 'DONE':
        break
    
for i in range(num_clients):
    client_agent_communicator.client_id = i
    client_agent_communicator.invoke_custom_action(action='close_connection', _client_id=i)
    
client_agent.clean_up()

end_time = time.time()
print(f"Total time taken: {end_time - start_time:.2f} seconds")