import argparse
from omegaconf import OmegaConf
from appfl.agent import ServerAgent
from appfl.comm.grpc import GRPCServerCommunicator, serve

argparser = argparse.ArgumentParser()
argparser.add_argument("--config", type=str, default="config.yaml")
args = argparser.parse_args()

# Load the configuration file
server_agent_config = OmegaConf.load(args.config)

# Create a server agent
server_agent = ServerAgent(server_agent_config=server_agent_config)

# Create a GRPC communicator using the server agent
communicator = GRPCServerCommunicator(
    server_agent,
    max_message_size=server_agent_config.server_configs.comm_configs.grpc_configs.max_message_size,
    logger=server_agent.logger,
)

# Start serving
serve(communicator, **server_agent_config.server_configs.comm_configs.grpc_configs)
