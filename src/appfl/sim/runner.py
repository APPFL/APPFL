from __future__ import annotations

import sys
import copy
import time

import torch
import numpy as np

from typing import Any
from omegaconf import DictConfig, OmegaConf

from appfl.sim.agent import ClientAgent
from appfl.logger import create_experiment_tracker
from appfl.sim.loaders import load_dataset, load_model
from appfl.sim.misc.system_utils import (
    _client_processing_chunk_size,
    _force_server_cpu_when_global_eval_disabled,
    _release_clients,
    get_local_rank,
    resolve_rank_device,
    set_seed_everything,
    validate_backend_device_consistency,
)
from appfl.sim.misc.config_utils import (
    _allow_reusable_on_demand_pool,
    _cfg_bool,
    _cfg_get,
    _cfg_to_dict,
    _default_config_path,
    _ensure_model_cfg,
    _merge_runtime_cfg,
    _parse_cli_tokens,
    _resolve_config_path,
)
from appfl.sim.misc.data_utils import (
    _apply_holdout_dataset_ratio,
    _dataset_has_eval_split,
    _sample_train_clients,
    _validate_algorithm_data_requirements,
    _validate_loader_output,
)
from appfl.sim.misc.learning_utils import (
    _aggregate_eval_stats,
    _adapt_bandit_policy,
    _build_federated_eval_plan,
    _run_federated_eval_serial,
    _should_eval_round,
    _weighted_global_stat,
)
from appfl.sim.misc.logging_utils import (
    _emit_client_state_policy_message,
    _emit_federated_eval_policy_message,
    _emit_logging_policy_message,
    _log_round,
    _new_server_logger,
    _resolve_run_log_dir,
    _resolve_run_id,
    _start_summary_lines,
    _warn_if_workers_pinned_to_single_device,
)
from appfl.sim.misc.runtime_utils import (
    _build_clients,
    _build_on_demand_worker_pool,
    _build_server,
    _collect_local_training_payload,
    _payload_to_updates,
    _select_round_local_steps,
    _resolve_runtime_policies,
)
from appfl.sim.misc.dist_utils import (
    _broadcast_model_state_inplace,
    _load_dataset_distributed,
    _run_federated_eval_distributed,
    launch_or_run_distributed,
)


def _assert_stateful_dataloaders_unchanged(
    persistent_clients: list[ClientAgent] | None,
    stateful_dataloader_ids: dict[int, tuple[int, int, int]] | None,
) -> None:
    if stateful_dataloader_ids is None or persistent_clients is None:
        return
    for client in persistent_clients:
        current_ids = (
            id(client.trainer.train_dataloader),
            id(client.trainer.val_dataloader)
            if client.trainer.val_dataloader is not None
            else -1,
            id(client.trainer.test_dataloader)
            if client.trainer.test_dataloader is not None
            else -1,
        )
        if current_ids != stateful_dataloader_ids[int(client.id)]:
            raise RuntimeError(
                "Stateful mode requires persistent per-client dataloaders across rounds."
            )


def _adapt_and_track_gen_reward(
    *,
    server,
    round_pre_val_error: float | None,
    prev_pre_val_error: float | None,
    track_gen_rewards: bool,
    cumulative_gen_reward: float,
) -> tuple[float | None, float | None, float]:
    round_gen_reward: float | None = None
    next_prev = prev_pre_val_error
    next_cumulative = float(cumulative_gen_reward)
    if round_pre_val_error is None:
        return round_gen_reward, next_prev, next_cumulative
    _adapt_bandit_policy(server, pre_val_error=round_pre_val_error)
    if track_gen_rewards and prev_pre_val_error is not None:
        round_gen_reward = float(prev_pre_val_error - round_pre_val_error)
        next_cumulative += float(round_gen_reward)
    next_prev = float(round_pre_val_error)
    return round_gen_reward, next_prev, next_cumulative


def _log_round_metrics(
    *,
    config: DictConfig,
    round_idx: int,
    selected_count: int,
    train_client_count: int,
    stats: dict,
    round_local_steps: int | None,
    round_wall_time_sec: float | None,
    round_gen_error: float | None,
    global_eval_metrics,
    federated_eval_metrics,
    federated_eval_out_metrics,
    track_gen_rewards: bool,
    round_gen_reward: float | None,
    cumulative_gen_reward: float | None,
    server_logger,
    tracker,
) -> None:
    _log_round(
        config,
        round_idx,
        selected_count,
        train_client_count,
        stats,
        round_local_steps=round_local_steps,
        round_wall_time_sec=round_wall_time_sec,
        global_gen_error=round_gen_error,
        global_eval_metrics=global_eval_metrics,
        federated_eval_metrics=federated_eval_metrics,
        federated_eval_out_metrics=federated_eval_out_metrics,
        track_gen_rewards=bool(track_gen_rewards),
        round_gen_reward=round_gen_reward,
        cumulative_gen_reward=float(cumulative_gen_reward)
        if bool(track_gen_rewards) and cumulative_gen_reward is not None
        else None,
        logger=server_logger,
        tracker=tracker,
    )


def _run_federated_eval_serial_round(
    *,
    config: DictConfig,
    enable_federated_eval: bool,
    round_idx: int,
    num_rounds: int,
    train_client_ids: list[int],
    holdout_client_ids: list[int],
    persistent_clients: list[ClientAgent] | None,
    server,
    on_demand_model,
    model,
    client_datasets,
    client_device: str,
    on_demand_workers: dict,
):
    federated_eval_metrics = None
    federated_eval_out_metrics = None
    if not enable_federated_eval:
        return federated_eval_metrics, federated_eval_out_metrics

    plan = _build_federated_eval_plan(
        config=config,
        round_idx=round_idx,
        num_rounds=num_rounds,
        train_client_ids=train_client_ids,
        holdout_client_ids=holdout_client_ids,
    )
    if persistent_clients is not None:
        state = server.model.state_dict()
        if plan["scheme"] == "client":
            eval_out_set = set(plan["out_ids"])
            eval_out_stats = {}
            for client in persistent_clients:
                if client.id in eval_out_set:
                    client.load_parameters(state)
                    eval_out_stats[int(client.id)] = client.evaluate(
                        split="test",
                        offload_after=False,
                    )
            federated_eval_out_metrics = _aggregate_eval_stats(eval_out_stats)
        else:
            eval_set = set(plan["in_ids"])
            eval_stats = {}
            for client in persistent_clients:
                if client.id not in eval_set:
                    continue
                client.load_parameters(state)
                eval_stats[int(client.id)] = client.evaluate(
                    split="test",
                    offload_after=False,
                )
            federated_eval_metrics = _aggregate_eval_stats(eval_stats)
        return federated_eval_metrics, federated_eval_out_metrics

    if plan["scheme"] == "client":
        federated_eval_out_metrics = _run_federated_eval_serial(
            config=config,
            model=on_demand_model if on_demand_model is not None else model,
            client_datasets=client_datasets,
            device=client_device,
            global_state=server.model.state_dict(),
            eval_client_ids=list(plan["out_ids"]),
            round_idx=round_idx,
            eval_tag="fed-out",
            eval_split="test",
            eval_num_workers_override=on_demand_workers["eval"],
        )
    else:
        federated_eval_metrics = _run_federated_eval_serial(
            config=config,
            model=on_demand_model if on_demand_model is not None else model,
            client_datasets=client_datasets,
            device=client_device,
            global_state=server.model.state_dict(),
            eval_client_ids=list(plan["in_ids"]),
            round_idx=round_idx,
            eval_tag="fed",
            eval_split="test",
            eval_num_workers_override=on_demand_workers["eval"],
        )
    return federated_eval_metrics, federated_eval_out_metrics


def _run_federated_eval_distributed_round(
    *,
    dist,
    rank: int,
    config: DictConfig,
    enable_federated_eval: bool,
    round_idx: int,
    num_rounds: int,
    train_client_ids: list[int],
    holdout_client_ids: list[int],
    on_demand_model,
    model,
    client_datasets,
    client_device: str,
    next_global_state,
    world_size: int,
):
    plan_payload = [None]
    if rank == 0 and enable_federated_eval:
        plan_payload[0] = _build_federated_eval_plan(
            config=config,
            round_idx=round_idx,
            num_rounds=num_rounds,
            train_client_ids=train_client_ids,
            holdout_client_ids=holdout_client_ids,
        )
    dist.broadcast_object_list(plan_payload, src=0)
    plan = plan_payload[0]

    federated_eval_metrics = None
    federated_eval_out_metrics = None
    if not (enable_federated_eval and isinstance(plan, dict)):
        return federated_eval_metrics, federated_eval_out_metrics

    if plan["scheme"] == "client":
        federated_eval_out_metrics = _run_federated_eval_distributed(
            config=config,
            model=on_demand_model if on_demand_model is not None else model,
            client_datasets=client_datasets,
            device=client_device,
            global_state=next_global_state,
            eval_client_ids=list(plan["out_ids"]),
            rank=rank,
            world_size=world_size,
            eval_split="test",
        )
    else:
        federated_eval_metrics = _run_federated_eval_distributed(
            config=config,
            model=on_demand_model if on_demand_model is not None else model,
            client_datasets=client_datasets,
            device=client_device,
            global_state=next_global_state,
            eval_client_ids=list(plan["in_ids"]),
            rank=rank,
            world_size=world_size,
            eval_split="test",
        )
    return federated_eval_metrics, federated_eval_out_metrics


def run_serial(config) -> None:
    if not isinstance(config, DictConfig):
        config = OmegaConf.create(_cfg_to_dict(config))
    _validate_algorithm_data_requirements(config)

    set_seed_everything(int(_cfg_get(config, "experiment.seed", 42)))
    t0 = time.time()
    run_id = _resolve_run_id()
    run_log_dir = _resolve_run_log_dir(config, run_id)
    server_logger = _new_server_logger(config, mode="serial", run_id=run_id)
    tracker = create_experiment_tracker(config, run_id=run_id)

    loader_cfg = _cfg_to_dict(config)
    # Respect user-configured download policy in serial mode.
    loader_cfg["download"] = _cfg_bool(config, "dataset.download", True)
    loader_cfg["logger"] = server_logger
    client_datasets, server_dataset, dataset_meta = load_dataset(loader_cfg)
    client_datasets = _apply_holdout_dataset_ratio(
        client_datasets, config=config, logger=server_logger
    )

    runtime_cfg = _merge_runtime_cfg(config, dataset_meta)
    _validate_loader_output(client_datasets, runtime_cfg)
    policy = _resolve_runtime_policies(config, runtime_cfg)
    algorithm_components = policy["algorithm_components"]
    num_clients = int(policy["num_clients"])
    state_policy = policy["state_policy"]
    train_client_ids = policy["train_client_ids"]
    holdout_client_ids = policy["holdout_client_ids"]
    num_sampled_clients = int(policy["num_sampled_clients"])
    logging_policy = policy["logging_policy"]
    _emit_logging_policy_message(
        logging_policy, num_clients=num_clients, logger=server_logger
    )
    _emit_client_state_policy_message(state_policy, logger=server_logger)
    server_logger.info(
        "Algorithm wiring: "
        f"algorithm={algorithm_components['algorithm_name']} "
        f"aggregator={algorithm_components['aggregator_name']} "
        f"scheduler={algorithm_components['scheduler_name']} "
        f"trainer={algorithm_components['trainer_name']}"
    )
    base_workers = max(0, int(_cfg_get(config, "train.num_workers", 0)))
    on_demand_workers = {"train": base_workers, "eval": base_workers}
    enable_global_eval = _cfg_bool(
        config, "eval.enable_global_eval", True
    ) and _dataset_has_eval_split(server_dataset)
    _force_server_cpu_when_global_eval_disabled(
        config, enable_global_eval, logger=server_logger
    )
    enable_federated_eval = _cfg_bool(config, "eval.enable_federated_eval", True)
    track_gen_rewards = _cfg_bool(config, "logging.configs.track_gen_rewards", False)
    num_rounds = int(_cfg_get(config, "train.num_rounds", 20))
    eval_every = int(_cfg_get(config, "eval.every", 1))
    prev_pre_val_error: float | None = None
    cumulative_gen_reward: float = 0.0

    model = load_model(
        runtime_cfg,
        input_shape=tuple(runtime_cfg["input_shape"]),
        num_classes=int(runtime_cfg["num_classes"]),
    )

    server = _build_server(
        config=config,
        runtime_cfg=runtime_cfg,
        model=model,
        server_dataset=server_dataset,
        algorithm_components=algorithm_components,
    )

    client_device = resolve_rank_device(
        str(_cfg_get(config, "experiment.device", "cpu")), rank=1, world_size=2
    )
    chunk_size = _client_processing_chunk_size(
        config=config,
        model=model,
        device=client_device,
        total_clients=num_clients,
        phase="train",
    )
    persistent_clients = None
    worker_pool: list[ClientAgent] | None = None
    stateful_mode = bool(state_policy["stateful"])
    on_demand_model = None if stateful_mode else copy.deepcopy(model)

    if stateful_mode:
        persistent_clients = _build_clients(
            config=config,
            model=model,
            client_datasets=client_datasets,
            local_client_ids=np.arange(num_clients).astype(int),
            device=client_device,
            run_log_dir=run_log_dir,
            client_logging_enabled=bool(logging_policy["client_logging_enabled"]),
            trainer_name=str(algorithm_components["trainer_name"]),
        )
    else:
        if _allow_reusable_on_demand_pool(
            config,
            client_logging_enabled=bool(logging_policy["client_logging_enabled"]),
        ):
            worker_pool = _build_on_demand_worker_pool(
                config=config,
                model=on_demand_model if on_demand_model is not None else model,
                client_datasets=client_datasets,
                local_client_ids=np.arange(num_clients).astype(int),
                device=client_device,
                run_log_dir=run_log_dir,
                client_logging_enabled=bool(logging_policy["client_logging_enabled"]),
                trainer_name=str(algorithm_components["trainer_name"]),
                pool_size=chunk_size,
                num_workers_override=on_demand_workers["train"],
            )
    if stateful_mode:
        if persistent_clients is None or worker_pool is not None:
            raise RuntimeError(
                "Invalid client lifecycle: experiment.stateful=true requires persistent clients."
            )
    stateful_dataloader_ids = None
    if stateful_mode and persistent_clients is not None:
        stateful_dataloader_ids = {
            int(client.id): (
                id(client.trainer.train_dataloader),
                id(client.trainer.val_dataloader)
                if client.trainer.val_dataloader is not None
                else -1,
                id(client.trainer.test_dataloader)
                if client.trainer.test_dataloader is not None
                else -1,
            )
            for client in persistent_clients
        }

    server_logger.info(
        _start_summary_lines(
            mode="serial",
            config=config,
            num_clients=num_clients,
            train_client_count=len(train_client_ids),
            holdout_client_count=len(holdout_client_ids),
            num_sampled_clients=num_sampled_clients,
        )
    )
    _emit_federated_eval_policy_message(
        config=config,
        train_client_count=len(train_client_ids),
        holdout_client_count=len(holdout_client_ids),
        logger=server_logger,
    )

    interrupted = False
    try:
        for round_idx in range(1, num_rounds + 1):
            round_t0 = time.time()
            selected_ids = _sample_train_clients(
                train_client_ids=train_client_ids,
                num_sampled_clients=int(num_sampled_clients),
            )
            round_local_steps = _select_round_local_steps(server, round_idx)
            global_state = server.model.state_dict()

            _assert_stateful_dataloaders_unchanged(
                persistent_clients=persistent_clients,
                stateful_dataloader_ids=stateful_dataloader_ids,
            )
            local_payload = _collect_local_training_payload(
                selected_client_ids=selected_ids,
                persistent_clients=persistent_clients,
                worker_pool=worker_pool,
                config=config,
                model=model,
                on_demand_model=on_demand_model,
                client_datasets=client_datasets,
                client_device=client_device,
                run_log_dir=run_log_dir,
                client_logging_enabled=bool(logging_policy["client_logging_enabled"]),
                trainer_name=str(algorithm_components["trainer_name"]),
                round_idx=round_idx,
                round_local_steps=round_local_steps,
                global_state=global_state,
                chunk_size=chunk_size,
                num_workers_override=on_demand_workers["train"],
            )
            updates, sample_sizes, stats = _payload_to_updates(local_payload)

            server.aggregate(
                updates,
                sample_sizes,
                client_train_stats=stats,
            )
            round_gen_error = _weighted_global_stat(
                stats=stats,
                sample_sizes=sample_sizes,
                stat_key="local_gen_error",
            )
            round_pre_val_error = _weighted_global_stat(
                stats=stats,
                sample_sizes=sample_sizes,
                stat_key="pre_val_loss",
            )
            round_gen_reward, prev_pre_val_error, cumulative_gen_reward = (
                _adapt_and_track_gen_reward(
                    server=server,
                    round_pre_val_error=round_pre_val_error,
                    prev_pre_val_error=prev_pre_val_error,
                    track_gen_rewards=bool(track_gen_rewards),
                    cumulative_gen_reward=float(cumulative_gen_reward),
                )
            )
            global_eval_metrics = None
            if enable_global_eval and _should_eval_round(
                round_idx,
                eval_every,
                num_rounds,
            ):
                global_eval_metrics = server.evaluate(round_idx=round_idx)

            federated_eval_metrics, federated_eval_out_metrics = (
                _run_federated_eval_serial_round(
                    config=config,
                    enable_federated_eval=enable_federated_eval,
                    round_idx=round_idx,
                    num_rounds=num_rounds,
                    train_client_ids=train_client_ids,
                    holdout_client_ids=holdout_client_ids,
                    persistent_clients=persistent_clients,
                    server=server,
                    on_demand_model=on_demand_model,
                    model=model,
                    client_datasets=client_datasets,
                    client_device=client_device,
                    on_demand_workers=on_demand_workers,
                )
            )

            _log_round_metrics(
                config=config,
                round_idx=round_idx,
                selected_count=len(selected_ids),
                train_client_count=len(train_client_ids),
                stats=stats,
                round_local_steps=round_local_steps,
                round_wall_time_sec=(time.time() - round_t0),
                round_gen_error=round_gen_error,
                global_eval_metrics=global_eval_metrics,
                federated_eval_metrics=federated_eval_metrics,
                federated_eval_out_metrics=federated_eval_out_metrics,
                track_gen_rewards=bool(track_gen_rewards),
                round_gen_reward=round_gen_reward,
                cumulative_gen_reward=float(cumulative_gen_reward),
                server_logger=server_logger,
                tracker=tracker,
            )
    except KeyboardInterrupt:
        interrupted = True
        server_logger.info("Interrupted by user; shutting down serial backend.")
    finally:
        if persistent_clients is not None:
            _release_clients(persistent_clients)
        if worker_pool is not None:
            _release_clients(worker_pool)
        if tracker is not None:
            tracker.close()

    if interrupted:
        raise SystemExit(130)
    server_logger.info(f"Finished serial simulation in {time.time() - t0:.2f}s.")
    server_logger.info("Saved resulting metrics in a log folder.")
    server_logger.info("Good Bye!")


def run_distributed(config, backend: str) -> None:
    import torch.distributed as dist

    if not isinstance(config, DictConfig):
        config = OmegaConf.create(_cfg_to_dict(config))
    _validate_algorithm_data_requirements(config)
    if backend not in {"nccl", "gloo"}:
        raise ValueError("backend must be one of: nccl, gloo")
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed process group is not initialized.")

    rank = int(dist.get_rank())
    world_size = int(dist.get_world_size())
    local_rank = get_local_rank(default=rank)

    if backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("backend=nccl requires CUDA.")
        torch.cuda.set_device(local_rank % max(1, torch.cuda.device_count()))

    set_seed_everything(int(_cfg_get(config, "experiment.seed", 42)) + rank)
    run_id_root = _resolve_run_id() if rank == 0 else ""
    run_id_payload = [run_id_root]
    dist.broadcast_object_list(run_id_payload, src=0)
    run_id = str(run_id_payload[0])
    run_log_dir = _resolve_run_log_dir(config, run_id)

    bootstrap_logger = (
        _new_server_logger(config, mode=f"{backend}-rank{rank}", run_id=run_id)
        if rank == 0
        else None
    )
    client_datasets, server_dataset, dataset_meta = _load_dataset_distributed(
        config=config,
        rank=rank,
        logger=bootstrap_logger if rank == 0 else None,
    )
    client_datasets = _apply_holdout_dataset_ratio(
        client_datasets, config=config, logger=bootstrap_logger if rank == 0 else None
    )

    runtime_cfg = _merge_runtime_cfg(config, dataset_meta)
    _validate_loader_output(client_datasets, runtime_cfg)
    policy = _resolve_runtime_policies(config, runtime_cfg)
    algorithm_components = policy["algorithm_components"]
    num_clients = int(policy["num_clients"])
    state_policy = policy["state_policy"]
    train_client_ids = policy["train_client_ids"]
    holdout_client_ids = policy["holdout_client_ids"]
    num_sampled_clients = int(policy["num_sampled_clients"])
    logging_policy = policy["logging_policy"]
    base_workers = max(0, int(_cfg_get(config, "train.num_workers", 0)))
    on_demand_workers = {"train": base_workers, "eval": base_workers}
    enable_global_eval = _cfg_bool(
        config, "eval.enable_global_eval", True
    ) and _dataset_has_eval_split(server_dataset)
    enable_federated_eval = _cfg_bool(config, "eval.enable_federated_eval", True)
    track_gen_rewards = _cfg_bool(config, "logging.configs.track_gen_rewards", False)
    num_rounds = int(_cfg_get(config, "train.num_rounds", 20))
    eval_every = int(_cfg_get(config, "eval.every", 1))
    model = load_model(
        runtime_cfg,
        input_shape=tuple(runtime_cfg["input_shape"]),
        num_classes=int(runtime_cfg["num_classes"]),
    )

    client_device = resolve_rank_device(
        str(_cfg_get(config, "experiment.device", "cpu")),
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
    )
    if backend == "nccl" and not str(client_device).startswith("cuda"):
        client_device = f"cuda:{local_rank % max(1, torch.cuda.device_count())}"

    client_groups = np.array_split(np.arange(num_clients), world_size)
    local_client_ids = np.asarray(client_groups[rank]).astype(int)
    local_client_set = {int(cid) for cid in local_client_ids}

    chunk_size = _client_processing_chunk_size(
        config=config,
        model=model,
        device=client_device,
        total_clients=max(1, len(local_client_ids)),
        phase="train",
    )
    persistent_clients = None
    worker_pool: list[ClientAgent] | None = None
    stateful_mode = bool(state_policy["stateful"])
    on_demand_model = None if stateful_mode else copy.deepcopy(model)
    local_client_logging_enabled = bool(logging_policy["client_logging_enabled"])

    if stateful_mode:
        persistent_clients = _build_clients(
            config=config,
            model=model,
            client_datasets=client_datasets,
            local_client_ids=local_client_ids,
            device=client_device,
            run_log_dir=run_log_dir,
            client_logging_enabled=local_client_logging_enabled,
            trainer_name=str(algorithm_components["trainer_name"]),
        )
    else:
        if _allow_reusable_on_demand_pool(
            config,
            client_logging_enabled=bool(logging_policy["client_logging_enabled"]),
        ):
            worker_pool = _build_on_demand_worker_pool(
                config=config,
                model=on_demand_model if on_demand_model is not None else model,
                client_datasets=client_datasets,
                local_client_ids=local_client_ids,
                device=client_device,
                run_log_dir=run_log_dir,
                client_logging_enabled=local_client_logging_enabled,
                trainer_name=str(algorithm_components["trainer_name"]),
                pool_size=chunk_size,
                num_workers_override=on_demand_workers["train"],
            )
    if stateful_mode:
        if persistent_clients is None or worker_pool is not None:
            raise RuntimeError(
                "Invalid client lifecycle: experiment.stateful=true requires persistent clients."
            )
    stateful_dataloader_ids = None
    if stateful_mode and persistent_clients is not None:
        stateful_dataloader_ids = {
            int(client.id): (
                id(client.trainer.train_dataloader),
                id(client.trainer.val_dataloader)
                if client.trainer.val_dataloader is not None
                else -1,
                id(client.trainer.test_dataloader)
                if client.trainer.test_dataloader is not None
                else -1,
            )
            for client in persistent_clients
        }

    tracker = None
    server = None
    server_logger = bootstrap_logger if rank == 0 else None
    prev_pre_val_error: float | None = None
    cumulative_gen_reward: float = 0.0
    if rank == 0:
        _force_server_cpu_when_global_eval_disabled(
            config, enable_global_eval, logger=server_logger
        )
        _warn_if_workers_pinned_to_single_device(
            config=config,
            world_size=world_size,
            logger=server_logger,
        )
        _emit_logging_policy_message(
            logging_policy, num_clients=num_clients, logger=server_logger
        )
        _emit_client_state_policy_message(state_policy, logger=server_logger)
        server_logger.info(
            "Algorithm wiring: "
            f"algorithm={algorithm_components['algorithm_name']} "
            f"aggregator={algorithm_components['aggregator_name']} "
            f"scheduler={algorithm_components['scheduler_name']} "
            f"trainer={algorithm_components['trainer_name']}"
        )
        server_logger.info(
            f"Distributed context: backend={backend} world_size={world_size} rank={rank} local_rank={local_rank}"
        )
        server_logger.info(
            _start_summary_lines(
                mode=backend,
                config=config,
                num_clients=num_clients,
                train_client_count=len(train_client_ids),
                holdout_client_count=len(holdout_client_ids),
                num_sampled_clients=num_sampled_clients,
            )
        )
        _emit_federated_eval_policy_message(
            config=config,
            train_client_count=len(train_client_ids),
            holdout_client_count=len(holdout_client_ids),
            logger=server_logger,
        )
        tracker = create_experiment_tracker(config, run_id=run_id)
        server = _build_server(
            config=config,
            runtime_cfg=runtime_cfg,
            model=model,
            server_dataset=server_dataset,
            algorithm_components=algorithm_components,
        )
    # Ensure every rank starts from the exact same global model state.
    sync_model = server.model if rank == 0 else model
    _broadcast_model_state_inplace(sync_model, src=0)
    if on_demand_model is not None and sync_model is not on_demand_model:
        on_demand_model.load_state_dict(sync_model.state_dict())

    t0 = time.time()
    interrupted = False
    try:
        for round_idx in range(1, num_rounds + 1):
            round_t0 = time.time() if rank == 0 else None
            if rank == 0:
                selected_ids = _sample_train_clients(
                    train_client_ids=train_client_ids,
                    num_sampled_clients=int(num_sampled_clients),
                )
                local_steps = _select_round_local_steps(server, round_idx)
                payload = {"selected_ids": selected_ids, "local_steps": local_steps}
            else:
                payload = None
            bcast_obj = [payload]
            dist.broadcast_object_list(bcast_obj, src=0)
            payload = bcast_obj[0]
            selected_ids = list(payload["selected_ids"])
            round_local_steps = payload.get("local_steps", None)
            sync_model = server.model if rank == 0 else model
            _broadcast_model_state_inplace(sync_model, src=0)
            if on_demand_model is not None and sync_model is not on_demand_model:
                on_demand_model.load_state_dict(sync_model.state_dict())
            global_state = sync_model.state_dict()

            selected_local_ids = sorted(
                int(cid) for cid in selected_ids if int(cid) in local_client_set
            )
            _assert_stateful_dataloaders_unchanged(
                persistent_clients=persistent_clients,
                stateful_dataloader_ids=stateful_dataloader_ids,
            )
            local_payload = _collect_local_training_payload(
                selected_client_ids=selected_local_ids,
                persistent_clients=persistent_clients,
                worker_pool=worker_pool,
                config=config,
                model=model,
                on_demand_model=on_demand_model,
                client_datasets=client_datasets,
                client_device=client_device,
                run_log_dir=run_log_dir,
                client_logging_enabled=local_client_logging_enabled,
                trainer_name=str(algorithm_components["trainer_name"]),
                round_idx=round_idx,
                round_local_steps=round_local_steps,
                global_state=global_state,
                chunk_size=chunk_size,
                num_workers_override=on_demand_workers["train"],
            )

            if rank == 0:
                gathered = [None] * world_size
                dist.gather_object(local_payload, object_gather_list=gathered, dst=0)
            else:
                dist.gather_object(local_payload, object_gather_list=None, dst=0)
                gathered = None
            stats = {}
            round_gen_reward = None
            round_gen_error = None
            if rank == 0:
                updates: dict[int, Any] = {}
                sample_sizes: dict[int, int] = {}
                stats = {}
                for payload_map in gathered or []:
                    if not isinstance(payload_map, dict):
                        continue
                    part_updates, part_sizes, part_stats = _payload_to_updates(
                        payload_map
                    )
                    updates.update(part_updates)
                    sample_sizes.update(part_sizes)
                    stats.update(part_stats)
                server.aggregate(
                    updates,
                    sample_sizes,
                    client_train_stats=stats,
                )
                round_gen_error = _weighted_global_stat(
                    stats=stats,
                    sample_sizes=sample_sizes,
                    stat_key="local_gen_error",
                )
                round_pre_val_error = _weighted_global_stat(
                    stats=stats,
                    sample_sizes=sample_sizes,
                    stat_key="pre_val_loss",
                )
                round_gen_reward, prev_pre_val_error, cumulative_gen_reward = (
                    _adapt_and_track_gen_reward(
                        server=server,
                        round_pre_val_error=round_pre_val_error,
                        prev_pre_val_error=prev_pre_val_error,
                        track_gen_rewards=bool(track_gen_rewards),
                        cumulative_gen_reward=float(cumulative_gen_reward),
                    )
                )

            sync_model = server.model if rank == 0 else model
            _broadcast_model_state_inplace(sync_model, src=0)
            if on_demand_model is not None and sync_model is not on_demand_model:
                on_demand_model.load_state_dict(sync_model.state_dict())
            next_global_state = sync_model.state_dict()

            global_eval_metrics = None
            if (
                rank == 0
                and enable_global_eval
                and _should_eval_round(
                    round_idx,
                    eval_every,
                    num_rounds,
                )
            ):
                global_eval_metrics = server.evaluate(round_idx=round_idx)

            federated_eval_metrics, federated_eval_out_metrics = (
                _run_federated_eval_distributed_round(
                    dist=dist,
                    rank=rank,
                    config=config,
                    enable_federated_eval=enable_federated_eval,
                    round_idx=round_idx,
                    num_rounds=num_rounds,
                    train_client_ids=train_client_ids,
                    holdout_client_ids=holdout_client_ids,
                    on_demand_model=on_demand_model,
                    model=model,
                    client_datasets=client_datasets,
                    client_device=client_device,
                    next_global_state=next_global_state,
                    world_size=world_size,
                )
            )
            if rank == 0:
                _log_round_metrics(
                    config=config,
                    round_idx=round_idx,
                    selected_count=len(selected_ids),
                    train_client_count=len(train_client_ids),
                    stats=stats,
                    round_local_steps=round_local_steps,
                    round_wall_time_sec=(
                        (time.time() - float(round_t0))
                        if isinstance(round_t0, (int, float))
                        else None
                    ),
                    round_gen_error=round_gen_error,
                    global_eval_metrics=global_eval_metrics,
                    federated_eval_metrics=federated_eval_metrics,
                    federated_eval_out_metrics=federated_eval_out_metrics,
                    track_gen_rewards=bool(track_gen_rewards),
                    round_gen_reward=round_gen_reward,
                    cumulative_gen_reward=float(cumulative_gen_reward),
                    server_logger=server_logger,
                    tracker=tracker,
                )
    except KeyboardInterrupt:
        interrupted = True
        if rank == 0 and server_logger is not None:
            server_logger.info(
                f"Interrupted by user; shutting down distributed backend ({backend})."
            )
    finally:
        if persistent_clients is not None:
            _release_clients(persistent_clients)
        if worker_pool is not None:
            _release_clients(worker_pool)
        if tracker is not None:
            tracker.close()
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except RuntimeError:
                pass

    if interrupted:
        raise SystemExit(130)
    if rank == 0 and server_logger is not None:
        server_logger.info(f"Finished {backend} simulation in {time.time() - t0:.2f}s.")
        server_logger.info("Saved resulting metrics in a log folder.")
        server_logger.info("Good Bye!")


def parse_config(argv: list[str] | None = None) -> tuple[str, DictConfig]:
    argv = list(sys.argv[1:] if argv is None else argv)
    config_path, backend_override, dotlist = _parse_cli_tokens(argv)

    default_path = _default_config_path()
    if not default_path.exists():
        raise FileNotFoundError(f"Default config not found: {default_path}")

    cfg = OmegaConf.load(default_path)
    if config_path:
        cfg_path = _resolve_config_path(config_path)
        cfg = OmegaConf.merge(cfg, OmegaConf.load(cfg_path))
    if dotlist:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
    _ensure_model_cfg(cfg)
    if backend_override is not None:
        cfg.experiment.backend = backend_override

    backend = str(_cfg_get(cfg, "experiment.backend", "serial")).lower()
    if backend not in {"serial", "nccl", "gloo"}:
        raise ValueError("backend must be one of: serial, nccl, gloo")

    return backend, cfg


def main(argv: list[str] | None = None) -> None:
    cli_argv = list(sys.argv[1:] if argv is None else argv)
    backend, config = parse_config(cli_argv)
    validate_backend_device_consistency(backend=backend, config=config)
    if backend == "serial":
        run_serial(config)
    else:
        launch_or_run_distributed(
            backend=backend, config=config, entry_fn=run_distributed
        )


if __name__ == "__main__":
    welcome_message = r"""
                  _____  _____  ______ _          _____ _____ __  __
            /\   |  __ \|  __ \|  ____| |        / ____|_   _|  \/  |
           /  \  | |__) | |__) | |__  | |  _____| (___   | | | \  / |
          / /\ \ |  ___/|  ___/|  __| | | |______\___ \  | | | |\/| |
         / ____ \| |    | |    | |    | |____    ____) |_| |_| |  | |
        /_/    \_\_|    |_|    |_|    |______|  |_____/|_____|_|  |_|

    Copyright © 2022-2026, UChicago Argonne, LLC and the APPFL Development Team
    """
    print(welcome_message)
    main()
