import socket
import argparse
from omegaconf import OmegaConf
from appfl.agent import ServerAgent
from appfl.communicator.grpc import GRPCServerCommunicator, serve

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--config", 
    type=str, 
    default="configs/communication/server_grpc_cnn.yaml",
    help="Path to the configuration file."
)
argparser.add_argument(
    "--num_clients",
    type=int,
    required=True,
)
argparser.add_argument(
    "--epochs",
    type=int,
    required=True,
)

args = argparser.parse_args()

def get_local_ip():
    try:
        # Create a dummy socket to help determine the local IP
        # The actual connection is not made
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))  # Google's DNS server
            ip = s.getsockname()[0]
    except Exception as e:
        ip = "Unable to determine IP: " + str(e)
    return ip

server_agent_config = OmegaConf.load(args.config)
server_agent_config.server_configs.scheduler_kwargs.num_clients = args.num_clients
server_agent_config.server_configs.num_global_epochs = args.epochs
server_agent_config.server_configs.comm_configs.grpc_configs.server_uri = f'{get_local_ip()}:50051'

server_agent = ServerAgent(server_agent_config=server_agent_config)

server_agent.logger.info(f"Starting gRPC server at {server_agent_config.server_configs.comm_configs.grpc_configs.server_uri} ...")
server_agent.logger.info(f"Total number of clients is {server_agent_config.server_configs.scheduler_kwargs.num_clients}, and the total epoch is {server_agent_config.server_configs.num_global_epochs}")

communicator = GRPCServerCommunicator(
    server_agent,
    max_message_size=server_agent_config.server_configs.comm_configs.grpc_configs.max_message_size,
    logger=server_agent.logger,
)

serve(
    communicator,
    **server_agent_config.server_configs.comm_configs.grpc_configs,
)
