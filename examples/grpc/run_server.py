import argparse
from omegaconf import OmegaConf
from appfl.agent import ServerAgent
from appfl.communicator.grpc import GRPCServerCommunicator, serve

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--config", 
    type=str, 
    default="./resources/configs/mnist/server_fedavg.yaml",
    help="Path to the configuration file."
)
args = argparser.parse_args()

server_agent_config = OmegaConf.load(args.config)
server_agent = ServerAgent(server_agent_config=server_agent_config)

communicator = GRPCServerCommunicator(
    server_agent,
    max_message_size=server_agent_config.server_configs.comm_configs.grpc_configs.max_message_size,
    logger=server_agent.logger,
)

serve(
    communicator,
    **server_agent_config.server_configs.comm_configs.grpc_configs,
)
