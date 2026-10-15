"""
Launcher for the Evo2 PEFT federated client. Identical to run_client.py,
plus sending the trainer's AIDRIN/DRISEE readiness report to the server.

Usage:
    python run_client_evo2.py --config ./resources/configs/evo2_wastewater_peft/client_1.yaml
"""

import argparse

from omegaconf import OmegaConf

from appfl.agent.client import ClientAgent
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

    client_config = client_comm.get_configuration()
    client_agent.load_config(client_config)  # triggers trainer __init__ + AIDRIN check

    aidrin_report = getattr(getattr(client_agent, "trainer", None), "_aidrin_report", {})

    init_global_model = client_comm.get_global_model(init_model=True)
    client_agent.load_parameters(init_global_model)

    if aidrin_report:
        flat = {
            "site": aidrin_report.get("site", ""),
            "n_clean": aidrin_report.get("n_clean", 0),
            "n_contaminated": aidrin_report.get("n_contaminated", 0),
            "site_ready": aidrin_report.get("site_ready", True),
            "drisee_threshold": aidrin_report.get("drisee_threshold", 20.0),
        }
        for k, v in aidrin_report.get("csv_meta_metrics", {}).items():
            if not k.startswith("_"):
                flat[f"csv_{k}"] = v
        try:
            # "get_data_readiness_report" is a reserved appfl action name;
            # use a distinct one.
            client_comm.invoke_custom_action(action="report_aidrin_readiness", **flat)
            logger.info("[AIDRIN] Data readiness report sent to server.")
        except Exception as e:
            logger.warning(f"[AIDRIN] Could not send readiness report to server: {e}")

    while True:
        client_agent.train()
        local_model = client_agent.get_parameters()
        if isinstance(local_model, tuple):
            local_model, metadata = local_model[0], local_model[1]
        else:
            metadata = {}
        new_global_model, metadata = client_comm.update_global_model(
            local_model, **metadata
        )
        if metadata["status"] == "DONE":
            break
        client_agent.load_parameters(new_global_model)

    client_comm.invoke_custom_action(action="close_connection")


if __name__ == "__main__":
    main()
