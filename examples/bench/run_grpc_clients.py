"""
Run multiple gRPC clients to benchmark the server.
"""

import time
import argparse
from mpi4py import MPI
from omegaconf import OmegaConf
from appfl.agent import ClientAgent
from appfl.communicator.grpc import GRPCClientCommunicator

argparser = argparse.ArgumentParser()
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

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
num_clients = size

# Set the client configurations
client_agent_config = OmegaConf.load(args.client_config)
client_agent_config.train_configs.logging_id = f'Client{rank}'
client_agent_config.data_configs.dataset_kwargs.num_clients = num_clients
client_agent_config.data_configs.dataset_kwargs.client_id = rank
client_agent_config.data_configs.dataset_kwargs.visualization = True if rank == 0 else False

if args.server_uri:
    client_agent_config.comm_configs.grpc_configs.server_uri = args.server_uri

client_agent = ClientAgent(client_agent_config=client_agent_config)
client_communicator = GRPCClientCommunicator(
    client_id = client_agent.get_id(),
    **client_agent_config.comm_configs.grpc_configs,
)

client_config = client_communicator.get_configuration()
client_agent.load_config(client_config)

init_global_model = client_communicator.get_global_model(init_model=True)
client_agent.load_parameters(init_global_model)

start_time = time.time()

while True:
    client_agent.train()
    local_model = client_agent.get_parameters()
    new_global_model, metadata = client_communicator.update_global_model(local_model)
    if metadata['status'] == 'DONE':
        break
    if 'local_steps' in metadata:
        client_agent.trainer.train_configs.num_local_steps = metadata['local_steps']
    client_agent.load_parameters(new_global_model)
client_communicator.invoke_custom_action(action='close_connection')
client_agent.clean_up()

end_time = time.time()

if rank == 0:
    print(f"Total time taken: {end_time - start_time:.2f} seconds")