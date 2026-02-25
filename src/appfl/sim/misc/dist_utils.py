from __future__ import annotations

import os
import socket
import time
from typing import Callable

import torch
from omegaconf import DictConfig, OmegaConf
from appfl.sim.misc.config_utils import _cfg_get
from appfl.sim.misc.config_utils import build_loss_from_config
from appfl.sim.metrics import parse_metric_names
from appfl.sim.misc.data_utils import _resolve_client_eval_dataset
from appfl.sim.misc.learning_utils import (
    _aggregate_eval_stats,
    _evaluate_dataset_direct,
)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _resolve_distributed_world_size(config: DictConfig, backend: str) -> int:
    if backend == "nccl":
        gpus = int(torch.cuda.device_count())
        if gpus <= 0:
            raise RuntimeError(
                "backend=nccl requested, but no CUDA devices are available."
            )
        return gpus
    cpu_slots = max(1, (os.cpu_count() or 2) - 1)
    configured_clients = int(_cfg_get(config, "train.num_clients", 0))
    if configured_clients > 0:
        return max(1, min(cpu_slots, configured_clients))
    return cpu_slots


def _distributed_worker_entry(
    rank: int,
    world_size: int,
    backend: str,
    config_dict: dict,
    master_port: int,
    entry_fn: Callable[[DictConfig, str], None],
) -> None:
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    try:
        entry_fn(OmegaConf.create(config_dict), backend)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def launch_or_run_distributed(
    backend: str,
    config: DictConfig,
    entry_fn: Callable[[DictConfig, str], None],
) -> None:
    import torch.distributed as dist
    import torch.multiprocessing as mp

    if dist.is_available() and dist.is_initialized():
        entry_fn(config, backend)
        return

    env_world_size = os.environ.get("WORLD_SIZE", "").strip()
    env_rank = os.environ.get("RANK", "").strip()
    if env_world_size and env_rank:
        dist.init_process_group(backend=backend, init_method="env://")
        try:
            entry_fn(config, backend)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
        return

    world_size = _resolve_distributed_world_size(config, backend)
    if world_size <= 0:
        raise RuntimeError("Unable to resolve distributed world size.")

    master_port = _find_free_port()
    ctx = mp.get_context("spawn")
    processes = []
    for rank in range(world_size):
        proc = ctx.Process(
            target=_distributed_worker_entry,
            args=(
                rank,
                world_size,
                backend,
                dict(OmegaConf.to_container(config, resolve=True)),
                master_port,
                entry_fn,
            ),
        )
        proc.start()
        processes.append(proc)

    def _terminate_processes(procs):
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=2.0)
        for p in procs:
            if p.is_alive():
                p.kill()

    try:
        while processes:
            alive = []
            failed_exit = None
            for proc in processes:
                proc.join(timeout=0.2)
                if proc.exitcode is None:
                    alive.append(proc)
                elif proc.exitcode != 0:
                    failed_exit = int(proc.exitcode)
            if failed_exit is not None:
                _terminate_processes(alive)
                raise RuntimeError(
                    f"Distributed worker exited with code {failed_exit}."
                )
            processes = alive
            if processes:
                time.sleep(0.05)
    except KeyboardInterrupt:
        _terminate_processes(processes)
        raise SystemExit(130)
    except Exception:
        _terminate_processes(processes)
        raise


def _load_dataset_distributed(
    config: DictConfig,
    rank: int,
    logger=None,
):
    import torch.distributed as dist
    from appfl.sim.loaders import load_dataset
    from appfl.sim.misc.config_utils import _cfg_to_dict

    def _set_download(cfg_dict: dict, value: bool) -> None:
        ds = cfg_dict.get("dataset", None)
        if not isinstance(ds, dict):
            ds = {}
            cfg_dict["dataset"] = ds
        ds["download"] = bool(value)

    loader_cfg = _cfg_to_dict(config)
    if logger is not None:
        loader_cfg["logger"] = logger
    if rank == 0:
        cfg_root = dict(loader_cfg)
        _set_download(cfg_root, True)
        cached = load_dataset(cfg_root)
    else:
        cached = None
    dist.barrier()
    if rank == 0:
        return cached
    cfg_other = dict(loader_cfg)
    _set_download(cfg_other, False)
    return load_dataset(cfg_other)


def _rank_client_span(
    total_clients: int, rank: int, world_size: int
) -> tuple[int, int]:
    start = (int(rank) * int(total_clients)) // int(world_size)
    end = ((int(rank) + 1) * int(total_clients)) // int(world_size)
    return start, end


def _gather_to_rank0(payload, *, rank: int, world_size: int):
    import torch.distributed as dist

    if rank == 0:
        gathered = [None] * int(world_size)
        dist.gather_object(payload, object_gather_list=gathered, dst=0)
        return gathered
    dist.gather_object(payload, object_gather_list=None, dst=0)
    return None


def _run_federated_eval_distributed(
    config: DictConfig,
    model,
    client_datasets,
    device: str,
    global_state,
    eval_client_ids: list[int],
    rank: int,
    world_size: int,
    eval_split: str = "test",
):
    if not eval_client_ids:
        _gather_to_rank0({}, rank=rank, world_size=world_size)
        return None

    eval_model = model.to(device)
    eval_model.load_state_dict(global_state)
    eval_model.eval()
    eval_loss_fn = build_loss_from_config(config)
    if hasattr(eval_loss_fn, "to"):
        eval_loss_fn = eval_loss_fn.to(device)
    eval_metric_names = parse_metric_names(_cfg_get(config, "eval.metrics", ["acc1"]))
    eval_batch_size = int(
        _cfg_get(
            config, "train.eval_batch_size", _cfg_get(config, "train.batch_size", 32)
        )
    )
    eval_workers = max(0, int(_cfg_get(config, "train.num_workers", 0)))
    start, end = _rank_client_span(
        total_clients=len(client_datasets),
        rank=rank,
        world_size=world_size,
    )
    local_client_set = set(range(int(start), int(end)))
    local_stats = {}
    for client_id in sorted(int(cid) for cid in eval_client_ids):
        if client_id not in local_client_set:
            continue
        eval_ds = _resolve_client_eval_dataset(
            client_datasets=client_datasets,
            client_id=int(client_id),
            eval_split=str(eval_split),
        )
        local_stats[int(client_id)] = _evaluate_dataset_direct(
            model=eval_model,
            dataset=eval_ds,
            device=device,
            loss_fn=eval_loss_fn,
            eval_metric_names=eval_metric_names,
            batch_size=eval_batch_size,
            num_workers=eval_workers,
        )

    gathered = _gather_to_rank0(local_stats, rank=rank, world_size=world_size)
    if rank != 0:
        return None

    merged = {}
    for payload in gathered or []:
        if isinstance(payload, dict):
            merged.update(payload)
    return _aggregate_eval_stats(merged)


def _broadcast_model_state_inplace(model, *, src: int = 0) -> None:
    import torch.distributed as dist

    state = model.state_dict()
    backend = str(dist.get_backend()).strip().lower()
    for key in sorted(state.keys()):
        tensor = state[key]
        if not torch.is_tensor(tensor):
            continue
        if backend == "nccl" and tensor.device.type == "cpu":
            device = torch.device("cuda", torch.cuda.current_device())
            with torch.no_grad():
                staged = tensor.detach().to(device=device, non_blocking=True)
                dist.broadcast(staged, src=src)
                tensor.copy_(staged.to(device="cpu", non_blocking=False))
            del staged
            continue
        dist.broadcast(tensor, src=src)
