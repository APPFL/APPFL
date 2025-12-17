import argparse
import threading
from omegaconf import OmegaConf
from appfl.agent import DFLNodeAgent
from appfl.comm.grpc import (
    GRPCDFLNodeServeCommunicator,
    GRPCDFLNodeConnectCommunicator,
    serve,
)

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--config",
    type=str,
    default="./configs/node_0.yaml",
    help="Path to the configuration file.",
)
args = argparser.parse_args()

dfl_node_agent_config = OmegaConf.load(args.config)
dfl_node_agent = DFLNodeAgent(dfl_node_agent_config=dfl_node_agent_config)

# Start serving the node
server_communicator = GRPCDFLNodeServeCommunicator(
    dfl_node_agent,
    max_message_size=dfl_node_agent_config.comm_configs.grpc_configs.server.max_message_size,
    logger=dfl_node_agent.logger,
)

server_thread = threading.Thread(
    target=serve,
    args=(server_communicator,),
    kwargs=dfl_node_agent_config.comm_configs.grpc_configs.server,
)
server_thread.start()

# Connect to the neighbor models
connect_communicators = [
    GRPCDFLNodeConnectCommunicator(
        node_id=dfl_node_agent.get_id(),
        **neighbor_kwargs,
    )
    for neighbor_kwargs in dfl_node_agent_config.comm_configs.grpc_configs.connect
]

# Start training
for epoch in range(dfl_node_agent_config.num_epochs):
    dfl_node_agent.train()
    neighbor_models = []
    for connect_communicator in connect_communicators:
        neighbor_models.append(connect_communicator.get_neighbor_model())
    dfl_node_agent.aggregate_parameters(neighbor_models=neighbor_models)


# Close the connection with the neighbors
for connect_communicator in connect_communicators:
    connect_communicator.invoke_custom_action(action="close_connection")

# Stop serving the node
server_thread.join()  # vaseline555: hang due to incomplete closing of connections..
