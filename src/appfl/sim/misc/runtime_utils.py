from __future__ import annotations
import copy
import importlib
import logging
from typing import TYPE_CHECKING, Any, Sequence
import numpy as np
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from appfl.sim.misc.config_utils import (
    _build_train_cfg,
    _cfg_bool,
    _cfg_to_dict,
    _cfg_get,
    _resolve_algorithm_components,
    _resolve_client_logging_policy,
)
from appfl.sim.misc.config_utils import build_loss_from_config
from appfl.sim.misc.data_utils import (
    _build_client_groups,
    _normalize_client_tuple,
    _resolve_num_sampled_clients,
)
from appfl.sim.misc.system_utils import _iter_id_chunks, _release_clients

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from appfl.sim.agent import ClientAgent, ServerAgent


def _select_round_local_steps(server, round_idx: int):
    scheduler = getattr(server, "scheduler", None)
    if scheduler is None or not hasattr(scheduler, "pull"):
        return None
    try:
        return int(scheduler.pull(round_idx=int(round_idx)))
    except (TypeError, ValueError, AttributeError) as exc:
        LOGGER.debug("Scheduler.pull failed to provide integer local steps: %s", exc)
        return None


def _run_local_client_update(
    client,
    *,
    global_state,
    round_idx: int,
    round_local_steps: int | None,
):
    client.load_parameters(global_state)
    if round_local_steps is None:
        train_result = client.train(round=round_idx)
    else:
        train_result = client.train(round=round_idx, local_steps=int(round_local_steps))
    uploaded = client.get_parameters()
    state = uploaded[0] if isinstance(uploaded, tuple) else uploaded
    return train_result, state


def _collect_local_training_payload(
    *,
    selected_client_ids: Sequence[int],
    persistent_clients: Sequence | None,
    worker_pool: Sequence | None,
    config: DictConfig,
    model,
    on_demand_model,
    client_datasets: Sequence,
    client_device: str,
    run_log_dir: str,
    client_logging_enabled: bool,
    trainer_name: str,
    round_idx: int,
    round_local_steps: int | None,
    global_state,
    chunk_size: int,
    num_workers_override: int | None,
) -> dict[int, dict[str, Any]]:
    local_payload: dict[int, dict[str, Any]] = {}
    if persistent_clients is not None:
        selected_set = {int(cid) for cid in selected_client_ids}
        for client in persistent_clients:
            if int(client.id) not in selected_set:
                continue
            train_result, state = _run_local_client_update(
                client,
                global_state=global_state,
                round_idx=round_idx,
                round_local_steps=round_local_steps,
            )
            local_payload[int(client.id)] = {
                "state": state,
                "num_examples": int(train_result.get("num_examples", 0)),
                "stats": train_result,
            }
        return local_payload

    for chunk_ids in _iter_id_chunks(selected_client_ids, chunk_size):
        if worker_pool:
            chunk_clients = worker_pool[: len(chunk_ids)]
            for client, cid in zip(chunk_clients, chunk_ids):
                _rebind_client_for_on_demand_job(
                    client,
                    client_id=int(cid),
                    client_datasets=client_datasets,
                    num_workers_override=num_workers_override,
                )
        else:
            chunk_clients = _build_clients(
                config=config,
                model=on_demand_model if on_demand_model is not None else model,
                client_datasets=client_datasets,
                local_client_ids=np.asarray(chunk_ids).astype(int),
                device=client_device,
                run_log_dir=run_log_dir,
                client_logging_enabled=client_logging_enabled,
                trainer_name=trainer_name,
                share_model=True,
                num_workers_override=num_workers_override,
            )
        for client in chunk_clients:
            train_result, state = _run_local_client_update(
                client,
                global_state=global_state,
                round_idx=round_idx,
                round_local_steps=round_local_steps,
            )
            local_payload[int(client.id)] = {
                "state": state,
                "num_examples": int(train_result.get("num_examples", 0)),
                "stats": train_result,
            }
        if not worker_pool:
            _release_clients(chunk_clients)
    return local_payload


def _payload_to_updates(local_payload: dict[int, dict[str, Any]]):
    updates: dict[int, Any] = {}
    sample_sizes: dict[int, int] = {}
    stats: dict[int, dict[str, Any]] = {}
    for cid, payload_item in local_payload.items():
        state = payload_item.get("state")
        if isinstance(state, tuple):
            state = state[0]
        updates[int(cid)] = state
        sample_sizes[int(cid)] = int(payload_item.get("num_examples", 0))
        stats[int(cid)] = payload_item.get("stats", {})
    return updates, sample_sizes, stats


def _resolve_runtime_policies(
    config: DictConfig, runtime_cfg: dict[str, Any]
) -> dict[str, Any]:
    num_clients = int(runtime_cfg["num_clients"])
    algorithm_components = _resolve_algorithm_components(config)
    state_policy = {
        "stateful": bool(_cfg_bool(config, "experiment.stateful", False)),
        "source": "experiment.stateful",
    }
    train_client_ids, holdout_client_ids = _build_client_groups(config, num_clients)
    num_sampled_clients = _resolve_num_sampled_clients(
        config, num_clients=len(train_client_ids)
    )
    logging_policy = _resolve_client_logging_policy(
        config,
        num_clients=num_clients,
        num_sampled_clients=num_sampled_clients,
    )
    return {
        "algorithm_components": algorithm_components,
        "num_clients": num_clients,
        "state_policy": state_policy,
        "train_client_ids": train_client_ids,
        "holdout_client_ids": holdout_client_ids,
        "num_sampled_clients": num_sampled_clients,
        "logging_policy": logging_policy,
    }


def _create_aggregator_instance(
    aggregator_name: str,
    model: Any | None,
    aggregator_config: DictConfig,
    logger: Any | None = None,
):
    try:
        appfl_module = importlib.import_module("appfl.sim.algorithm.aggregator")
        AggregatorClass = getattr(appfl_module, aggregator_name)
        return AggregatorClass(model, aggregator_config, logger)
    except AttributeError:
        raise ValueError(f"Invalid aggregator name: {aggregator_name}")


def _create_scheduler_instance(
    scheduler_name: str,
    scheduler_config: DictConfig,
    aggregator: Any | None = None,
    logger: Any | None = None,
):
    try:
        appfl_module = importlib.import_module("appfl.sim.algorithm.scheduler")
        SchedulerClass = getattr(appfl_module, scheduler_name)
        return SchedulerClass(scheduler_config, aggregator, logger)
    except AttributeError:
        raise ValueError(f"Invalid scheduler name: {scheduler_name}")


def _rebind_client_for_on_demand_job(
    client: ClientAgent,
    *,
    client_id: int,
    client_datasets: Sequence,
    num_workers_override: int | None = None,
) -> None:
    train_ds, val_ds, test_ds = _normalize_client_tuple(client_datasets[int(client_id)])

    client.id = int(client_id)
    client.client_agent_config.client_id = str(int(client_id))
    client.train_dataset = train_ds
    client.val_dataset = val_ds
    client.test_dataset = test_ds

    trainer = getattr(client, "trainer", None)
    if trainer is None:
        return
    trainer.client_id = str(int(client_id))
    trainer.train_dataset = train_ds
    trainer.val_dataset = val_ds
    trainer.test_dataset = test_ds

    cfg = client.client_agent_config.train_configs
    if num_workers_override is None:
        num_workers = int(cfg.get("num_workers", 0))
    else:
        num_workers = max(0, int(num_workers_override))
    train_bs = int(cfg.get("batch_size", 32))
    val_bs = int(cfg.get("eval_batch_size", train_bs))
    train_shuffle = bool(cfg.get("train_data_shuffle", True))
    train_pin_memory = bool(cfg.get("train_pin_memory", False))
    eval_pin_memory = bool(cfg.get("eval_pin_memory", train_pin_memory))
    persistent_workers = bool(cfg.get("dataloader_persistent_workers", False))
    prefetch_factor = int(cfg.get("dataloader_prefetch_factor", 2))

    common_train_kwargs = {
        "batch_size": max(1, train_bs),
        "shuffle": train_shuffle,
        "num_workers": num_workers,
        "pin_memory": train_pin_memory,
    }
    common_eval_kwargs = {
        "batch_size": max(1, val_bs),
        "num_workers": num_workers,
        "pin_memory": eval_pin_memory,
    }
    if num_workers > 0:
        common_train_kwargs["persistent_workers"] = persistent_workers
        common_eval_kwargs["persistent_workers"] = persistent_workers
        common_train_kwargs["prefetch_factor"] = max(2, prefetch_factor)
        common_eval_kwargs["prefetch_factor"] = max(2, prefetch_factor)

    trainer.train_dataloader = DataLoader(
        train_ds,
        **common_train_kwargs,
    )
    trainer.val_dataloader = (
        DataLoader(
            val_ds,
            shuffle=False,
            **common_eval_kwargs,
        )
        if val_ds is not None
        else None
    )
    trainer.test_dataloader = (
        DataLoader(
            test_ds,
            shuffle=False,
            **common_eval_kwargs,
        )
        if test_ds is not None
        else None
    )


def _build_on_demand_worker_pool(
    config: DictConfig,
    model,
    client_datasets: Sequence,
    local_client_ids: Sequence[int],
    device: str,
    run_log_dir: str,
    client_logging_enabled: bool,
    trainer_name: str,
    pool_size: int,
    trainer_kwargs: dict[str, Any] | None = None,
    num_workers_override: int | None = None,
) -> list[ClientAgent]:
    if int(pool_size) <= 0:
        return []
    available_ids = [int(cid) for cid in local_client_ids]
    if not available_ids:
        return []
    ids: list[int] = []
    for idx in range(int(pool_size)):
        ids.append(int(available_ids[idx % len(available_ids)]))
    return _build_clients(
        config=config,
        model=model,
        client_datasets=client_datasets,
        local_client_ids=np.asarray(ids).astype(int),
        device=device,
        run_log_dir=run_log_dir,
        client_logging_enabled=client_logging_enabled,
        trainer_name=trainer_name,
        trainer_kwargs=trainer_kwargs,
        share_model=True,
        num_workers_override=num_workers_override,
    )


def _build_clients(
    config: DictConfig,
    model,
    client_datasets: Sequence,
    local_client_ids,
    device: str,
    run_log_dir: str,
    client_logging_enabled: bool = True,
    trainer_name: str = "FedavgTrainer",
    trainer_kwargs: dict[str, Any] | None = None,
    share_model: bool = False,
    num_workers_override: int | None = None,
):
    from appfl.sim.agent import ClientAgent

    train_cfg = _build_train_cfg(
        config,
        device=device,
        run_log_dir=run_log_dir,
        num_workers_override=num_workers_override,
    )
    if trainer_kwargs is None:
        trainer_kwargs = _cfg_get(config, "algorithm.trainer_kwargs", {})
    if trainer_kwargs:
        train_cfg.update(_cfg_to_dict(trainer_kwargs))
    train_cfg["client_logging_enabled"] = bool(client_logging_enabled)
    clients = []
    for cid in local_client_ids:
        train_ds, val_ds, test_ds = _normalize_client_tuple(client_datasets[int(cid)])
        client_cfg = OmegaConf.create(
            {
                "train_configs": {
                    **train_cfg,
                    "trainer": str(trainer_name),
                },
                "model_configs": {},
                "data_configs": {},
            }
        )
        client_cfg.client_id = str(int(cid))
        client_cfg.experiment_id = str(_cfg_get(config, "experiment.name", "appfl-sim"))
        client = ClientAgent(client_agent_config=client_cfg)
        client.model = model if share_model else copy.deepcopy(model)
        client.train_dataset = train_ds
        client.val_dataset = val_ds
        client.test_dataset = test_ds
        client.trainer = None
        client._load_trainer()
        client.id = int(cid)
        clients.append(client)
    return clients


def _build_server(
    config: DictConfig,
    runtime_cfg: dict,
    model,
    server_dataset,
    algorithm_components: dict[str, Any] | None = None,
) -> ServerAgent:
    from appfl.sim.agent import ServerAgent

    if algorithm_components is None:
        algorithm_components = _resolve_algorithm_components(config)
    num_clients = int(runtime_cfg["num_clients"])
    num_sampled_clients = _resolve_num_sampled_clients(config, num_clients=num_clients)
    loss_configs = _cfg_to_dict(_cfg_get(config, "loss.configs", {}))
    server_cfg = OmegaConf.create(
        {
            "client_configs": {
                "train_configs": {
                    "eval_metrics": _cfg_get(config, "eval.metrics", ["acc1"]),
                    "loss": {
                        "name": str(_cfg_get(config, "loss.name", "CrossEntropyLoss")),
                        "backend": str(_cfg_get(config, "loss.backend", "auto")),
                        "path": str(_cfg_get(config, "loss.path", "")),
                        "configs": loss_configs,
                    },
                },
                "model_configs": {},
            },
            "server_configs": {
                "num_clients": num_clients,
                "num_sampled_clients": int(num_sampled_clients),
                "device": str(_cfg_get(config, "experiment.server_device", "cpu")),
                "eval_show_progress": _cfg_bool(
                    config, "eval.show_eval_progress", True
                ),
                "eval_batch_size": int(
                    _cfg_get(
                        config,
                        "train.eval_batch_size",
                        _cfg_get(config, "train.batch_size", 32),
                    )
                ),
                "num_workers": int(_cfg_get(config, "train.num_workers", 0)),
                "eval_metrics": _cfg_get(config, "eval.metrics", ["acc1"]),
                "aggregator": str(algorithm_components["aggregator_name"]),
                "aggregator_kwargs": dict(algorithm_components["aggregator_kwargs"]),
                "scheduler": str(algorithm_components["scheduler_name"]),
                "scheduler_kwargs": {
                    **dict(algorithm_components["scheduler_kwargs"]),
                    "num_clients": num_clients,
                },
            },
        }
    )
    server = ServerAgent(server_agent_config=server_cfg)
    server.model = model
    if (
        hasattr(server, "aggregator")
        and server.aggregator is not None
        and hasattr(server.aggregator, "model")
        and getattr(server.aggregator, "model", None) is None
    ):
        server.aggregator.model = server.model
    server.loss_fn = build_loss_from_config(config)
    server._eval_dataset = server_dataset
    server._load_eval_data()
    return server
