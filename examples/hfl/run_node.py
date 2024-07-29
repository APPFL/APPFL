import argparse
from omegaconf import OmegaConf
from appfl.agent import HFLNodeAgent
from appfl.communicator.grpc import GRPCHFLNodeServeCommunicator, GRPCHFLNodeConnectCommunicator, serve

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--config", 
    type=str, 
    default="./configs/hfl/node.yaml",
    help="Path to the configuration file."
)
args = argparser.parse_args()

hfl_node_agent_config = OmegaConf.load(args.config)
hfl_node_agent = HFLNodeAgent(hfl_node_agent_config=hfl_node_agent_config)
connect_communicator = GRPCHFLNodeConnectCommunicator(
    node_id=hfl_node_agent.get_id(),
    **hfl_node_agent_config.comm_configs.grpc_configs.connect,
)

serve_communicator = GRPCHFLNodeServeCommunicator(
    hfl_node_agent,
    connect_communicator=connect_communicator,
    max_message_size=hfl_node_agent_config.comm_configs.grpc_configs.serve.max_message_size,
    logger=hfl_node_agent.logger,
)

serve(
    serve_communicator,
    **hfl_node_agent_config.comm_configs.grpc_configs.serve,
)
