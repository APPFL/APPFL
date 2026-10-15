"""
gRPC client for the setting where the server does not have the model definition.
Each client loads the model from its own config and starts training directly,
without requesting an initial global model from the server.

Usage (from the examples/ directory):
    python grpc/run_server.py --config ./resources/configs/mnist/server_fedavg_client_model.yaml
    python grpc/run_client_with_model.py --config ./resources/configs/mnist/client_1_with_model.yaml
    python grpc/run_client_with_model.py --config ./resources/configs/mnist/client_2_with_model.yaml
"""

import torch
import argparse
from omegaconf import OmegaConf
from appfl.agent import ClientAgent
from appfl.comm.grpc import GRPCClientCommunicator


argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--config",
    type=str,
    default="./resources/configs/mnist/client_1_with_model.yaml",
    help="Path to the configuration file.",
)
args = argparser.parse_args()

client_agent_config = OmegaConf.load(args.config)

# Seed before the model is created so that all clients start from the same initial weights,
# since there is no initial global model from the server to synchronize them.
torch.manual_seed(client_agent_config.model_configs.get("init_seed", 42))

client_agent = ClientAgent(client_agent_config=client_agent_config)
client_communicator = GRPCClientCommunicator(
    client_id=client_agent.get_id(),
    logger=client_agent.logger,
    **client_agent_config.comm_configs.grpc_configs,
)

client_config = client_communicator.get_configuration()
client_agent.load_config(client_config)

# Send the number of local data to the server
sample_size = client_agent.get_sample_size()
client_communicator.invoke_custom_action(
    action="set_sample_size", sample_size=sample_size
)

metadata = {}
while True:
    client_agent.train(**metadata)
    local_model = client_agent.get_parameters()
    if isinstance(local_model, tuple):
        local_model, metadata = local_model[0], local_model[1]
    else:
        metadata = {}
    new_global_model, metadata = client_communicator.update_global_model(
        local_model, **metadata
    )
    if metadata["status"] == "DONE":
        break
    if "local_steps" in metadata:
        client_agent.trainer.train_configs.num_local_steps = metadata["local_steps"]
    client_agent.load_parameters(new_global_model)
client_communicator.invoke_custom_action(action="close_connection")
