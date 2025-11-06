import argparse
from omegaconf import OmegaConf
from appfl.agent import ClientAgent
from appfl.comm.grpc import GRPCClientCommunicator


argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--config",
    type=str,
    default="./resources/configs/flamby/heart_disease/client_1.yaml",
    help="Path to the configuration file.",
)
args = argparser.parse_args()

client_agent_config = OmegaConf.load(args.config)

client_agent = ClientAgent(client_agent_config=client_agent_config)
client_communicator = GRPCClientCommunicator(
    client_id=client_agent.get_id(),
    logger=client_agent.logger,
    **client_agent_config.comm_configs.grpc_configs,
)

client_config = client_communicator.get_configuration()
client_agent.load_config(client_config)

# Generate data readiness report
if (
    hasattr(client_config, "data_readiness_configs")
    and hasattr(client_config.data_readiness_configs, "generate_dr_report")
    and client_config.data_readiness_configs.generate_dr_report
):
    # Check CADREModule availability and if the data needs remediation
    if (
        hasattr(client_config.data_readiness_configs.dr_metrics, "cadremodule_configs")
        and hasattr(
            client_config.data_readiness_configs.dr_metrics.cadremodule_configs,
            "remedy_action",
        )
        and client_config.data_readiness_configs.dr_metrics.cadremodule_configs.remedy_action
    ):
        client_agent.adapt_data(client_config=client_config)
    data_readiness = client_agent.generate_readiness_report(client_config)
    client_communicator.invoke_custom_action(
        action="get_data_readiness_report", **data_readiness
    )

client_communicator.invoke_custom_action(action="close_connection")
