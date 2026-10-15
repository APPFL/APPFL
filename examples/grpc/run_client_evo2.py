"""
Launcher for the Evo2 PEFT federated client (see
`appfl.algorithm.trainer.evo2_trainer.Evo2PEFTTrainer`).

Same flow as `run_client.py`, plus `grpc_retry`/`send_aidrin_report` from
`evo2_trainer` so the retry handling for transient gRPC errors against an
unreliable public endpoint (e.g. an APPFLx relay) doesn't have to be
reimplemented per launcher.

Usage:
    python run_client_evo2.py --config ./resources/configs/evo2_wastewater_peft/client_1.yaml
"""

import argparse

from omegaconf import OmegaConf

from appfl.agent.client import ClientAgent
from appfl.algorithm.trainer.evo2_trainer import grpc_retry, send_aidrin_report
from appfl.comm.grpc import GRPCClientCommunicator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", default="./resources/configs/evo2_wastewater_peft/client_1.yaml"
    )
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)

    client_agent = ClientAgent(client_agent_config=cfg)
    logger = client_agent.logger
    client_comm = GRPCClientCommunicator(
        client_id=client_agent.get_id(),
        logger=logger,
        **cfg.comm_configs.grpc_configs,
    )

    client_config = grpc_retry(client_comm.get_configuration, "get_configuration", logger)
    client_agent.load_config(client_config)  # triggers trainer __init__ + AIDRIN check

    aidrin_report = getattr(getattr(client_agent, "trainer", None), "_aidrin_report", {})

    result = grpc_retry(
        lambda: client_comm.get_global_model(init_model=True),
        "get_global_model(init)",
        logger,
    )
    init_model, metadata = result if isinstance(result, tuple) else (result, {})
    client_agent.load_parameters(init_model)

    send_aidrin_report(client_comm, aidrin_report, logger)

    while True:
        client_agent.train(**metadata)
        local_model = client_agent.get_parameters()
        local_model, metadata = (
            local_model if isinstance(local_model, tuple) else (local_model, {})
        )

        new_global_model, metadata = grpc_retry(
            lambda m=local_model, md=metadata: client_comm.update_global_model(m, **md),
            "update_global_model",
            logger,
        )
        if metadata["status"] == "DONE":
            break
        client_agent.load_parameters(new_global_model)

    client_comm.invoke_custom_action(action="close_connection")


if __name__ == "__main__":
    main()
