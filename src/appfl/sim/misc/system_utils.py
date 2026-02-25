from __future__ import annotations
import os
import random
import warnings
from typing import List, Optional, Sequence
import gc
import numpy as np
import torch
from omegaconf import DictConfig
from appfl.sim.logger import ServerAgentFileLogger
from appfl.sim.misc.config_utils import _cfg_get, _cfg_set

# Memory utilities are shared with the main appfl package.
from appfl.misc.memory_utils import (  # noqa: F401
    clone_state_dict_optimized,
    extract_model_state_optimized,
    safe_inplace_operation,
)



def set_seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _read_int_env(keys: List[str]) -> Optional[int]:
    for key in keys:
        value = os.environ.get(key, "")
        if value == "":
            continue
        try:
            return int(value)
        except Exception:
            continue
    return None


def get_local_rank(default: int = 0) -> int:
    """Best-effort local rank detection across common distributed launchers."""
    detected = _read_int_env(
        [
            "LOCAL_RANK",
            "OMPI_COMM_WORLD_LOCAL_RANK",
            "MV2_COMM_WORLD_LOCAL_RANK",
            "MPI_LOCALRANKID",
            "SLURM_LOCALID",
            "PMI_LOCAL_RANK",
        ]
    )
    if detected is None:
        return int(default)
    return max(0, int(detected))


def resolve_rank_device(
    base_device: str,
    rank: int,
    world_size: int,
    local_rank: Optional[int] = None,
) -> str:
    del world_size  # Reserved for future placement strategies.

    base = str(base_device).strip().lower()
    if not base.startswith("cuda") or not torch.cuda.is_available():
        return "cpu"

    if ":" in base:
        suffix = base.split(":", 1)[1].strip()
        if suffix and suffix.isdigit():
            return f"cuda:{int(suffix)}"
        if suffix not in {"", "local", "auto"}:
            return "cpu"

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        return "cpu"

    if local_rank is None:
        local_rank = get_local_rank(default=max(rank - 1, 0))
    gpu_idx = int(local_rank) % num_gpus
    return f"cuda:{gpu_idx}"


def validate_backend_device_consistency(backend: str, config: DictConfig) -> None:
    device = str(_cfg_get(config, "experiment.device", "cpu")).strip().lower()
    server_device = str(_cfg_get(config, "experiment.server_device", "cpu")).strip().lower()
    cuda_available = bool(torch.cuda.is_available())
    visible_gpus = int(torch.cuda.device_count()) if cuda_available else 0

    if server_device.startswith("cuda") and not cuda_available:
        raise ValueError(
            "server_device is CUDA but CUDA is unavailable. "
            "Set `experiment.server_device=cpu`."
        )

    if backend == "nccl":
        if not cuda_available:
            raise ValueError("backend=nccl requires CUDA, but CUDA is unavailable.")
        if not device.startswith("cuda"):
            raise ValueError("backend=nccl requires `device` to be CUDA (e.g., `cuda`).")
        if server_device.startswith("cuda"):
            if visible_gpus <= 2:
                warnings.warn(
                    "`server_device` is CUDA with backend=nccl and limited visible GPUs; "
                    "server may reduce client GPU memory headroom.",
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    "`server_device` is CUDA with backend=nccl; prefer `server_device=cpu` "
                    "unless server-side GPU eval is required.",
                    stacklevel=2,
                )
        return

    if backend == "gloo" and device.startswith("cuda"):
        warnings.warn(
            "`backend=gloo` with CUDA device is valid but unusual for FL simulation throughput. "
            "Use `backend=nccl` for multi-GPU CUDA runs.",
            stacklevel=2,
        )

    if backend == "serial" and device == "cpu" and cuda_available:
        warnings.warn(
            "CUDA is available but `device=cpu` in serial mode. "
            "Set `device=cuda` for faster local training if desired.",
            stacklevel=2,
        )


def _force_server_cpu_when_global_eval_disabled(
    config: DictConfig,
    enable_global_eval: bool,
    logger: Optional[ServerAgentFileLogger] = None,
) -> None:
    if enable_global_eval:
        return
    current = str(_cfg_get(config, "experiment.server_device", "cpu")).strip().lower()
    if not current.startswith("cuda"):
        return
    _cfg_set(config, "experiment.server_device", "cpu")
    msg = (
        "Global eval is disabled; forcing `server_device=cpu` to avoid unnecessary "
        "server-side GPU memory usage."
    )
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)


def _resolve_cuda_index(device: str) -> int:
    text = str(device).strip().lower()
    if not text.startswith("cuda"):
        return 0
    if ":" not in text:
        return 0
    suffix = text.split(":", 1)[1].strip()
    if suffix.isdigit():
        return int(suffix)
    return 0


def _model_bytes(model) -> int:
    total = 0
    for p in model.parameters():
        total += int(p.numel()) * int(p.element_size())
    for b in model.buffers():
        total += int(b.numel()) * int(b.element_size())
    return int(total)


def _client_processing_chunk_size(
    config: DictConfig,
    model=None,
    device: str = "cpu",
    total_clients: int = 0,
    phase: str = "train",
) -> int:
    phase_name = str(phase).strip().lower()
    if str(device).strip().lower().startswith("cuda") and torch.cuda.is_available():
        try:
            dev_idx = _resolve_cuda_index(device)
            free_bytes, _ = torch.cuda.mem_get_info(dev_idx)
        except Exception:
            free_bytes = 0
        model_bytes = _model_bytes(model) if model is not None else 64 * 1024 * 1024
        per_client = max(256 * 1024 * 1024, model_bytes * (10 if phase_name == "train" else 4))
        budget = int(float(free_bytes) * 0.35) if free_bytes > 0 else 0
        auto_chunk = int(budget // per_client) if budget > 0 else 1
        auto_chunk = max(1, min(64, auto_chunk))
    else:
        cpu = max(1, (os.cpu_count() or 1))
        auto_chunk = max(1, min(64, cpu // 2))

    if int(total_clients) > 0:
        auto_chunk = min(auto_chunk, int(total_clients))
    return max(1, int(auto_chunk))


def _iter_id_chunks(ids: Sequence[int], chunk_size: int):
    ordered = list(ids)
    for start in range(0, len(ordered), chunk_size):
        yield ordered[start : start + chunk_size]


def _release_clients(
    clients,
    clear_cuda_cache: bool = False,
    collect_garbage: bool = False,
) -> None:
    if clients is None:
        return
    if isinstance(clients, list):
        clients.clear()
    del clients
    if collect_garbage:
        gc.collect()
    if clear_cuda_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()
