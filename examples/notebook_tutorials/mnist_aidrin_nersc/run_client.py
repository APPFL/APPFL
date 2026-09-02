"""Script version of the client notebooks, for connecting to the notebook server.

Run the server notebook through its "Launch the server" cell, take the address it
prints, then start each client in its own terminal:

    python run_client.py --client_id 1 --server_uri 172.31.79.131:50051
    python run_client.py --client_id 2 --server_uri 172.31.79.131:50051

Both clients must be running: the readiness-report step and the synchronous
training rounds block until every client reaches them.

This does exactly what APPFL_Client{1,2}_MNIST_AIDRIN.ipynb do — inspect the local
dataset, apply the CADRE module remedy, send the readiness report, then train —
but prints the metrics instead of rendering them, and saves the plots as PNG files
next to this script when `--save_plots` is given.
"""

import os
import base64
import random
import argparse
import warnings

import torch
import numpy as np
from omegaconf import OmegaConf

from appfl.agent import ClientAgent
from appfl.comm.grpc import GRPCClientCommunicator

# Relative paths in the shared configs are written from `examples/`, two levels up
# from this file. Resolving it here lets the script run from any directory, rather
# than depending on the notebook's one-shot `os.chdir("../..")`.
EXAMPLES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TUTORIAL_DIR = os.path.dirname(os.path.abspath(__file__))


def set_seed(seed: int) -> None:
    """Seed every RNG the client draws from, matching the notebook."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def show_report(report, title, client_agent, save_plots=False, prefix=""):
    """
    Print the readiness metrics, and optionally write the plots to PNG files.

    :param report: Readiness report returned by `generate_readiness_report`.
    :param title: Heading for this report (e.g. "BEFORE remedy").
    :param client_agent: The client, used to report the input tensor shape.
    :param save_plots: Whether to write the base64 plots out as PNG files.
    :param prefix: Filename prefix for saved plots.
    """
    print(f"\n================ {title} ================")
    for key, value in report.items():
        if key in ("plots", "to_combine", "specified_metrics"):
            continue
        print(f"  {key:>22}: {value}")
    if report.get("specified_metrics"):
        print(f"  {'CADRE module metric':>22}: {report['specified_metrics']}")
    input_shape = tuple(client_agent.train_dataset[0][0].shape)
    print(f"  {'input tensor shape':>22}: {input_shape}")
    print("=" * (34 + len(title)))

    plots = report.get("plots", {})
    if not plots:
        return
    if not save_plots:
        print(f"  ({len(plots)} plot(s) omitted; pass --save_plots to write them)")
        return
    for plot_name, encoded in plots.items():
        path = os.path.join(TUTORIAL_DIR, f"{prefix}{plot_name}.png")
        with open(path, "wb") as f:
            f.write(base64.b64decode(encoded))
        print(f"  saved plot: {path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--client_id",
        type=int,
        choices=(1, 2),
        required=True,
        help="which client to run; selects client_N_cadremodule.yaml",
    )
    parser.add_argument(
        "--server_uri",
        type=str,
        required=True,
        help="address printed by the server notebook, e.g. 172.31.79.131:50051",
    )
    parser.add_argument(
        "--use_ssl",
        action="store_true",
        help="enable TLS; needed for a hosted endpoint such as *.trac.appflx.link",
    )
    parser.add_argument(
        "--max_message_size",
        type=int,
        default=None,
        help="raise this if the readiness report trips a gRPC message-size error; "
        "the server must be given the same value",
    )
    parser.add_argument(
        "--save_plots",
        action="store_true",
        help="write the readiness plots as PNG files next to this script",
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    os.chdir(EXAMPLES_DIR)
    set_seed(args.seed)

    # ---- client agent ----
    config_path = (
        f"./resources/configs/mnist_dr/client_{args.client_id}_cadremodule.yaml"
    )
    client_agent_config = OmegaConf.load(config_path)
    grpc_configs = client_agent_config.comm_configs.grpc_configs
    grpc_configs["server_uri"] = args.server_uri
    grpc_configs["use_ssl"] = args.use_ssl
    if args.max_message_size is not None:
        grpc_configs["max_message_size"] = args.max_message_size

    client_agent = ClientAgent(client_agent_config=client_agent_config)
    print(f"Client ID           : {client_agent.get_id()}")
    print(f"Local training size : {client_agent.get_sample_size()}")
    print(f"Connecting to       : {args.server_uri}")

    client_communicator = GRPCClientCommunicator(
        client_id=client_agent.get_id(),
        logger=client_agent.logger,
        **grpc_configs,
    )

    # ---- readiness configuration comes from the server ----
    client_config = client_communicator.get_configuration()
    cadremodule_configs = (
        client_config.data_readiness_configs.dr_metrics.cadremodule_configs
    )
    print(f"\nCADRE module: {cadremodule_configs.cadremodule_path}")

    prefix = f"client{args.client_id}_"
    report_before = client_agent.generate_readiness_report(client_config)
    show_report(
        report_before,
        "BEFORE remedy",
        client_agent,
        args.save_plots,
        prefix + "before_",
    )

    # ---- apply the remedy ----
    if cadremodule_configs.get("remedy_action", False):
        size_before = client_agent.get_sample_size()
        client_agent.adapt_data(client_config)
        size_after = client_agent.get_sample_size()
        print(
            f"\nCADRE remedy applied: {size_before} -> {size_after} local training samples"
        )
    else:
        print("\nremedy_action is False: reporting only, local dataset left untouched.")

    report_after = client_agent.generate_readiness_report(client_config)
    show_report(
        report_after, "AFTER remedy", client_agent, args.save_plots, prefix + "after_"
    )

    # ---- send the report, then train ----
    client_communicator.invoke_custom_action(
        action="set_sample_size", sample_size=client_agent.get_sample_size()
    )
    client_communicator.invoke_custom_action(
        action="get_data_readiness_report", **report_after
    )
    print(
        "\nData readiness report sent. The merged HTML report is written by the server."
    )

    client_agent.load_config(client_config)
    init_global_model = client_communicator.get_global_model(init_model=True)
    if isinstance(init_global_model, tuple):
        init_global_model, metadata = init_global_model
    else:
        metadata = {}
    client_agent.load_parameters(init_global_model)

    while True:
        client_agent.train(**metadata)
        local_model = client_agent.get_parameters()
        if isinstance(local_model, tuple):
            local_model, metadata = local_model
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
    print("Federated learning finished.")


if __name__ == "__main__":
    main()
