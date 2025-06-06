import argparse
from omegaconf import OmegaConf
from resources.cadre_module import CADREModule
from appfl.comm.grpc import GRPCClientCommunicator

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--config",
    type=str,
    default="./resources/configs/mnist/client_1.yaml",
    help="Path to the configuration file.",
)
args = argparser.parse_args()

client_agent_config = OmegaConf.load(args.config)

client_agent = CADREModule(client_agent_config=client_agent_config)
client_communicator = GRPCClientCommunicator(
    client_id=client_agent.get_id(),
    **client_agent_config.comm_configs.grpc_configs,
)

client_config = client_communicator.get_configuration()
client_agent.load_config(client_config)

# Generate the readiness report
data_readiness = client_agent.generate_mnist_readiness_report()
client_communicator.invoke_custom_action(
    action="get_mnist_readiness_report", **data_readiness
)

init_global_model = client_communicator.get_global_model(init_model=True)
client_agent.load_parameters(init_global_model)

while True:
    client_agent.train()
    local_model = client_agent.get_parameters()
    if isinstance(local_model, tuple):
        local_model, meta_data_local = local_model[0], local_model[1]
    else:
        meta_data_local = {}
    new_global_model, metadata = client_communicator.update_global_model(
        local_model, **meta_data_local
    )
    if metadata["status"] == "DONE":
        break
    client_agent.load_parameters(new_global_model)
client_communicator.invoke_custom_action(action="close_connection")
