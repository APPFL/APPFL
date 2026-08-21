"""
MPI run script for federated CLIP + LoRA fine-tuning with SVD aggregation.

Each client fine-tunes a CLIP model with LoRA adapters locally.
The server aggregates LoRA parameters via AB_SVD (FederatedLoRASVDAggregator),
producing a single global model broadcast to all clients each round.

Dataset, shots, and data distribution are fully configured in the client
YAML — swap dataset_name in dataset_kwargs to change datasets without
touching this script.

Usage:
    mpirun --oversubscribe --bind-to none -n 6 python mpi/run_mpi_decaf.py \\
        --server_config resources/configs/cifar10/server_decaf.yaml \\
        --client_config resources/configs/cifar10/client_decaf_iid.yaml

Command-line arguments:
    --server_config    Path to server YAML config.
    --client_config    Path to shared client YAML config template.
    --seed             Global random seed for reproducibility. Default: 42.
"""

import argparse
import os
import random

import numpy as np
import torch
from mpi4py import MPI
from omegaconf import OmegaConf

from appfl.agent import ClientAgent, ServerAgent
from appfl.comm.mpi import MPIClientCommunicator, MPIServerCommunicator

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Federated CLIP + LoRA fine-tuning with SVD aggregation"
)
parser.add_argument(
    "--server_config",
    type=str,
    default="./resources/configs/cifar10/server_decaf.yaml",
    help="Path to server configuration YAML.",
)
parser.add_argument(
    "--client_config",
    type=str,
    default="./resources/configs/cifar10/client_decaf_iid.yaml",
    help="Path to client configuration YAML (shared template).",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Global random seed for reproducibility.",
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Build results output path
# Structure: results/<config_stem>/clients<N>_rnd<R>_ls<LS>_r<R>_lr<LR>_seed<S>/
# ---------------------------------------------------------------------------

_config_stem = os.path.splitext(os.path.basename(args.server_config))[0]
if _config_stem.startswith("server_"):
    _config_stem = _config_stem[len("server_") :]

_cfg = OmegaConf.load(args.server_config)
_tc = _cfg.get("client_configs", {}).get("train_configs", {})
_mk = _cfg.get("client_configs", {}).get("model_configs", {}).get("model_kwargs", {})
_sc = _cfg.get("server_configs", {})

_rounds = _sc.get("num_global_epochs", "?")
_ls = _tc.get("num_local_steps", "?")
_lora_r = _mk.get("r", "?")
_lr_raw = _tc.get("lr", "?")


def _fmt_lr(v):
    try:
        f = float(v)
        return f"{f:.0e}".replace("e-0", "e-").replace("e+0", "e")
    except Exception:
        return str(v)


_num_clients = MPI.COMM_WORLD.Get_size() - 1

OUTPUT_DIR = os.path.join(
    "results",
    _config_stem,
    (
        f"clients{_num_clients}"
        f"_rnd{_rounds}"
        f"_ls{_ls}"
        f"_r{_lora_r}"
        f"_lr{_fmt_lr(_lr_raw)}"
        f"_seed{args.seed}"
    ),
)

# ---------------------------------------------------------------------------
# MPI setup + reproducibility seeding
# ---------------------------------------------------------------------------

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
num_clients = size - 1

_rank_seed = args.seed + rank
random.seed(_rank_seed)
np.random.seed(_rank_seed)
torch.manual_seed(_rank_seed)
torch.cuda.manual_seed_all(_rank_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ======================================================================
# Server process (rank 0)
# ======================================================================

if rank == 0:
    server_agent_config = OmegaConf.load(args.server_config)
    server_agent_config.server_configs.num_clients = num_clients
    server_agent_config.server_configs.logging_output_dirname = OUTPUT_DIR
    server_agent_config.client_configs.train_configs.logging_output_dirname = OUTPUT_DIR

    # Propagate lora_rank from model_kwargs.r to aggregator_kwargs so the
    # SVD truncation rank always matches the client LoRA rank.
    model_kwargs = server_agent_config.client_configs.model_configs.get(
        "model_kwargs", {}
    )
    if model_kwargs and "r" in model_kwargs:
        server_agent_config.server_configs.aggregator_kwargs.lora_rank = model_kwargs.r

    server_agent = ServerAgent(server_agent_config=server_agent_config)
    server_communicator = MPIServerCommunicator(
        comm, server_agent, logger=server_agent.logger
    )
    server_communicator.serve()

# ======================================================================
# Client processes (ranks 1 … N)
# ======================================================================

else:
    client_agent_config = OmegaConf.load(args.client_config)
    client_agent_config.client_id = f"Client{rank}"
    client_agent_config.train_configs.logging_output_dirname = OUTPUT_DIR

    # Standard FL overrides: num_clients and client_id drive data partitioning
    client_agent_config.data_configs.dataset_kwargs.num_clients = num_clients
    client_agent_config.data_configs.dataset_kwargs.client_id = rank - 1

    client_agent = ClientAgent(client_agent_config=client_agent_config)
    client_communicator = MPIClientCommunicator(
        comm, server_rank=0, client_id=client_agent_config.client_id
    )

    # Bootstrap: load config, initial model, register sample size
    client_config = client_communicator.get_configuration()
    client_agent.load_config(client_config)

    init_global_model = client_communicator.get_global_model(init_model=True)
    client_agent.load_parameters(init_global_model)

    sample_size = client_agent.get_sample_size()
    client_communicator.invoke_custom_action(
        action="set_sample_size", sample_size=sample_size
    )

    # Federated training loop
    while True:
        client_agent.train()

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
