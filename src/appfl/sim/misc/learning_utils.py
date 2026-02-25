from __future__ import annotations
import logging
from typing import Dict, List, Optional
import numpy as np
import torch
from omegaconf import DictConfig
from appfl.sim.metrics import MetricsManager, parse_metric_names
from appfl.sim.misc.config_utils import _cfg_bool
from appfl.sim.misc.config_utils import build_loss_from_config
from appfl.sim.misc.data_utils import _resolve_client_eval_dataset
from appfl.sim.misc.metrics_utils import _weighted_mean
from appfl.sim.misc.system_utils import _client_processing_chunk_size, _iter_id_chunks
from appfl.sim.misc.logging_utils import _new_progress
from appfl.sim.misc.config_utils import _cfg_get

LOGGER = logging.getLogger(__name__)


def _should_eval_round(round_idx: int, every: int, num_rounds: int) -> bool:
    every_i = int(every)
    if every_i <= 0:
        # Non-positive cadence disables periodic checkpoints and keeps only final-round eval.
        return round_idx == int(num_rounds)
    return round_idx % every_i == 0 or round_idx == int(num_rounds)

def _weighted_global_stat(
    stats: dict,
    sample_sizes: dict,
    stat_key: str,
) -> float | None:
    if not stats:
        return None
    total = 0.0
    accum = 0.0
    for cid, client_stats in stats.items():
        if not isinstance(client_stats, dict):
            continue
        value = client_stats.get(stat_key, None)
        if not isinstance(value, (int, float)):
            continue
        weight = float(sample_sizes.get(cid, 0))
        if weight <= 0.0:
            weight = 1.0
        accum += weight * float(value)
        total += weight
    if total <= 0.0:
        return None
    return float(accum / total)


def _adapt_bandit_policy(server, pre_val_error: float):
    scheduler = getattr(server, "scheduler", None)
    if scheduler is None or not hasattr(scheduler, "adapt"):
        return None
    try:
        return scheduler.adapt(pre_val_error=float(pre_val_error))
    except (TypeError, ValueError, AttributeError) as exc:
        LOGGER.debug("Scheduler.adapt failed for pre_val_error=%s: %s", pre_val_error, exc)
        return None

def _resolve_model_output(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (list, tuple)) and len(output) > 0:
        first = output[0]
        if torch.is_tensor(first):
            return first
    if isinstance(output, dict):
        for key in ("logits", "predictions", "output"):
            value = output.get(key, None)
            if torch.is_tensor(value):
                return value
        for value in output.values():
            if torch.is_tensor(value):
                return value
    raise TypeError("Model output is not a tensor-like object.")

def _evaluate_dataset_direct(
    model,
    dataset,
    device: str,
    loss_fn,
    eval_metric_names: List[str],
    batch_size: int,
    num_workers: int,
) -> Dict[str, float]:
    if dataset is None:
        return {"loss": -1.0, "num_examples": 0, "metrics": {}}
    try:
        n = len(dataset)
    except Exception:
        n = 0
    if int(n) <= 0:
        return {"loss": -1.0, "num_examples": 0, "metrics": {}}

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(num_workers)),
    )
    manager = MetricsManager(eval_metrics=eval_metric_names)
    total_examples = 0
    for data, target in loader:
        data = data.to(device)
        target = target.to(device)
        logits = _resolve_model_output(model(data))
        loss = loss_fn(logits, target)

        logits_cpu = logits.detach().cpu()
        target_cpu = target.detach().cpu()
        manager.track(float(loss.item()), logits_cpu, target_cpu)
        batch = int(target_cpu.shape[0]) if target_cpu.ndim > 0 else 1
        total_examples += batch

    stats = manager.aggregate(total_len=total_examples)
    return stats

def _build_federated_eval_plan(
    config: DictConfig,
    round_idx: int,
    num_rounds: int,
    train_client_ids: List[int],
    holdout_client_ids: List[int],
) -> Dict[str, List[int] | str | bool]:
    scheme = str(_cfg_get(config, "eval.configs.scheme", "dataset")).strip().lower()
    checkpoint = _should_eval_round(
        round_idx,
        int(_cfg_get(config, "eval.every", 1)),
        num_rounds,
    )

    if not checkpoint:
        return {
            "scheme": "client" if scheme == "client" else "dataset",
            "checkpoint": False,
            "in_ids": [],
            "out_ids": [],
        }

    if scheme == "client":
        out_ids = sorted(int(cid) for cid in holdout_client_ids)
        return {
            "scheme": "client",
            "checkpoint": checkpoint,
            "in_ids": [],
            "out_ids": out_ids,
        }

    # Default: dataset-based evaluation.
    # Evaluate all train clients at checkpoint rounds.
    in_ids = sorted(int(cid) for cid in train_client_ids)
    return {
        "scheme": "dataset",
        "checkpoint": checkpoint,
        "in_ids": in_ids,
        "out_ids": [],
    }

def _aggregate_eval_stats(stats: Dict[int, Dict]) -> Optional[Dict[str, float]]:
    if not stats:
        return None
    total_examples = sum(int(v.get("num_examples", 0)) for v in stats.values())
    numeric_keys = sorted(
        {
            key
            for values in stats.values()
            for key, value in values.items()
            if isinstance(value, (int, float))
            and key not in {"num_examples", "num_clients"}
            and not key.endswith("_std")
            and not key.endswith("_min")
            and not key.endswith("_max")
        }
    )

    result: Dict[str, float] = {
        "num_examples": int(max(total_examples, 0)),
        "num_clients": int(len(stats)),
    }

    if total_examples <= 0:
        for key in numeric_keys:
            result[key] = -1.0
            result[f"{key}_std"] = -1.0
            result[f"{key}_min"] = -1.0
            result[f"{key}_max"] = -1.0
        if "loss" not in result:
            result["loss"] = -1.0
        result.setdefault("loss_std", -1.0)
        result.setdefault("loss_min", -1.0)
        result.setdefault("loss_max", -1.0)
        return result

    for key in numeric_keys:
        values = [
            float(client_stats[key])
            for client_stats in stats.values()
            if key in client_stats and isinstance(client_stats.get(key), (int, float))
        ]
        if not values:
            continue
        result[key] = float(_weighted_mean(stats, key))
        result[f"{key}_std"] = float(np.std(values))
        result[f"{key}_min"] = float(min(values))
        result[f"{key}_max"] = float(max(values))

    if "loss" not in result:
        result["loss"] = -1.0
    result.setdefault("loss_std", 0.0)
    result.setdefault("loss_min", result["loss"])
    result.setdefault("loss_max", result["loss"])
    return result

def _run_federated_eval_serial(
    config: DictConfig,
    model,
    client_datasets,
    device: str,
    global_state,
    eval_client_ids: List[int],
    round_idx: int,
    eval_tag: str = "federated",
    eval_split: str = "test",
    eval_num_workers_override: Optional[int] = None,
) -> Optional[Dict[str, float]]:
    if not eval_client_ids:
        return None
    eval_loss_fn = build_loss_from_config(config)
    eval_metric_names = parse_metric_names(_cfg_get(config, "eval.metrics", ["acc1"]))
    eval_batch_size = int(
        _cfg_get(config, "train.eval_batch_size", _cfg_get(config, "train.batch_size", 32))
    )
    eval_workers = (
        int(_cfg_get(config, "train.num_workers", 0))
        if eval_num_workers_override is None
        else max(0, int(eval_num_workers_override))
    )
    model.load_state_dict(global_state)
    model = model.to(device)
    if hasattr(eval_loss_fn, "to"):
        eval_loss_fn = eval_loss_fn.to(device)
    was_training = bool(getattr(model, "training", False))
    model.eval()
    chunk_size = _client_processing_chunk_size(
        config=config,
        model=model,
        device=device,
        total_clients=len(eval_client_ids),
        phase="eval",
    )
    eval_stats: Dict[int, Dict] = {}
    progress = _new_progress(
        total=len(eval_client_ids),
        desc=f"Server (Round {int(round_idx):04d}) | Evaluation ({str(eval_tag).replace('-', ' ').title()}.)",
        enabled=_cfg_bool(config, "eval.show_eval_progress", True),
    )
    try:
        for chunk_ids in _iter_id_chunks(sorted(eval_client_ids), chunk_size):
            for client_id in chunk_ids:
                eval_ds = _resolve_client_eval_dataset(
                    client_datasets=client_datasets,
                    client_id=int(client_id),
                    eval_split=str(eval_split),
                )
                eval_stats[int(client_id)] = _evaluate_dataset_direct(
                    model=model,
                    dataset=eval_ds,
                    device=device,
                    loss_fn=eval_loss_fn,
                    eval_metric_names=eval_metric_names,
                    batch_size=eval_batch_size,
                    num_workers=eval_workers,
                )
            if progress is not None:
                progress.update(len(chunk_ids))
    finally:
        if was_training:
            model.train()
        if progress is not None:
            progress.close()
    return _aggregate_eval_stats(eval_stats)
