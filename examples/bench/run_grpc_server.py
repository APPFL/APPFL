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
    '--server_uri',
    type=str,
    required=False,
)
args = argparser.parse_args()

server_agent_config = OmegaConf.load(args.config)
server_agent = ServerAgent(server_agent_config=server_agent_config)

communicator = GRPCServerCommunicator(
    server_agent,
    max_message_size=server_agent_config.server_configs.comm_configs.grpc_configs.max_message_size,
    logger=server_agent.logger,
)

if args.server_uri:
    server_agent_config.server_configs.comm_configs.grpc_configs.server_uri = args.server_uri

server_agent.logger.info(f"Starting gRPC server at {server_agent_config.server_configs.comm_configs.grpc_configs.server_uri} ...")

serve(
    communicator,
    **server_agent_config.server_configs.comm_configs.grpc_configs,
)
