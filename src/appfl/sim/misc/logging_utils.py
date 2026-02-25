from __future__ import annotations
import hashlib
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict
import numpy as np
from omegaconf import DictConfig
from appfl.sim.logger.server_logger import ServerAgentFileLogger
from appfl.metrics import parse_metric_names
from appfl.sim.misc.config_utils import _cfg_bool, _cfg_get
from appfl.logger.utils import _remap_server_wandb_payload  # noqa: F401

class _AvgStdMetric(TypedDict):
    avg: float
    std: float

class _ClientsSummary(TypedDict):
    selected: int
    total: int

class _PolicySummary(TypedDict):
    tau_t: int

class _TimingSummary(TypedDict):
    round_wall_time_sec: float

class _GenRewardSummary(TypedDict):
    round: Optional[float]
    cumulative: Optional[float]

class _MinMaxMetric(TypedDict):
    min: float
    max: float

class _RoundMetricsPayload(TypedDict, total=False):
    clients: _ClientsSummary
    policy: _PolicySummary
    timing: _TimingSummary
    training: Dict[str, _AvgStdMetric]
    local_pre_val: Dict[str, _AvgStdMetric]
    local_post_val: Dict[str, _AvgStdMetric]
    local_pre_test: Dict[str, _AvgStdMetric]
    local_post_test: Dict[str, _AvgStdMetric]
    local_gen_error: _AvgStdMetric
    global_gen_error: float
    gen_reward: _GenRewardSummary
    global_eval: Dict[str, float | _AvgStdMetric]
    fed_eval: Dict[str, float | _AvgStdMetric]
    fed_eval_in: Dict[str, float | _AvgStdMetric]
    fed_eval_out: Dict[str, float | _AvgStdMetric]
    fed_extrema: Dict[str, _MinMaxMetric]


try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None


def _sanitize_wandb_token(token: str) -> str:
    text = re.sub(r"[^0-9a-zA-Z_]+", "_", str(token).strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return text.lower()


def _format_wandb_panel_key(panel: str, parts: List[str]) -> str:
    tokens = [_sanitize_wandb_token(p) for p in parts if str(p).strip()]
    tokens = [t for t in tokens if t]
    leaf = "_".join(tokens) if tokens else "metric"
    return f"{panel}/{leaf}"


def _set_unique_wandb_metric(
    payload: Dict[str, float],
    *,
    key: str,
    value: float,
    source: str,
    source_by_key: Dict[str, str],
) -> str:
    candidate = str(key)
    existing_source = source_by_key.get(candidate, None)
    if candidate in payload and existing_source != source:
        digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:8]
        candidate = f"{key}__{digest}"
        idx = 2
        while candidate in payload and source_by_key.get(candidate) != source:
            candidate = f"{key}__{digest}_{idx}"
            idx += 1
    source_by_key[candidate] = str(source)
    payload[candidate] = float(value)
    return candidate


def _orchestration_client_wandb_key(client_id: str, metric: str) -> str:
    client_token = _sanitize_wandb_token(client_id) or "client"
    metric_token = _sanitize_wandb_token(metric) or "metric"
    return f"orchestration/{client_token}_{metric_token}"


def _resolve_run_dir_path(config: DictConfig, run_id: str) -> Path:
    run_name = str(
        _cfg_get(config, "logging.name", _cfg_get(config, "experiment.name", "appfl-sim"))
    )
    return (
        Path(str(_cfg_get(config, "logging.path", "./logs")))
        / str(_cfg_get(config, "experiment.name", "appfl-sim"))
        / run_name
        / str(run_id).strip()
    )


def _remap_server_wandb_payload(flat_payload: Dict[str, float]) -> Dict[str, float]:
    remapped: Dict[str, float] = {}
    remap_sources: Dict[str, str] = {}
    eval_roots = {
        "global_eval",
        "fed_eval",
        "fed_eval_in",
        "fed_eval_out",
        "fed_extrema",
        "local_pre_val",
        "local_post_val",
        "local_pre_test",
        "local_post_test",
    }
    orchestration_roots = {
        "clients",
        "policy",
        "timing",
        "gen_reward",
        "local_gen_error",
    }
    for raw_key, raw_value in flat_payload.items():
        parts = [p for p in str(raw_key).split("/") if p]
        if not parts:
            continue
        if len(parts) == 2 and parts[0] == "clients" and parts[1] in {"selected", "total"}:
            continue

        root = parts[0]
        if root == "training":
            dst = _format_wandb_panel_key("training", parts[1:])
        elif root in eval_roots:
            dst = _format_wandb_panel_key("evaluation", parts)
        elif root in orchestration_roots or root in {"global_gen_error", "round"}:
            dst = _format_wandb_panel_key("orchestration", parts)
        else:
            dst = _format_wandb_panel_key("orchestration", parts)
        _set_unique_wandb_metric(
            remapped,
            key=dst,
            value=float(raw_value),
            source=str(raw_key),
            source_by_key=remap_sources,
        )
    return remapped

def _new_progress(total: int, desc: str, enabled: bool):
    if not enabled or _tqdm is None or int(total) <= 0:
        return None
    label = f"appfl-sim: ✅[{str(desc)}]"
    return _tqdm(
        total=int(total),
        desc=label,
        leave=False,
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )

def _emit_logging_policy_message(
    policy: Dict[str, object],
    num_clients: int,
    logger: Optional[ServerAgentFileLogger] = None,
) -> None:
    requested = str(policy["requested_scheme"])
    effective = str(policy["effective_scheme"])
    basis_clients = int(policy.get("basis_clients", num_clients))
    total_clients = int(policy.get("total_clients", num_clients))
    forced_server_only = bool(policy.get("forced_server_only", False))

    def _info(msg: str) -> None:
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

    def _warn(msg: str) -> None:
        if logger is not None:
            logger.warning(msg)
        else:
            print(msg)

    if forced_server_only:
        _info(
            "Per-client file logging disabled because `num_sampled_clients` < `num_clients`. "
            "Using `server-only` logging for performance."
        )
        return

    if requested == "auto" and effective == "server_only":
        _info(
            "Client logging auto-switched to `server_only` for this run."
        )
        return
    if requested == "server_only":
        _info("Using `logging_scheme`=`server_only` (server-side metrics only).")
        return
    if requested == "both":
        _warn(
            f"Per-client logging is explicitly enabled with sampled_clients={basis_clients} "
            f"(total_clients={total_clients}). This may produce large I/O overhead."
        )

def _emit_client_state_policy_message(
    policy: Dict[str, object],
    logger: Optional[ServerAgentFileLogger] = None,
) -> None:
    stateful = bool(policy.get("stateful", False))
    mode = "stateful/persistent" if stateful else "stateless/sporadic"
    msg = f"{mode.title()} clients because `stateful={stateful}`"
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)

def _emit_federated_eval_policy_message(
    config: DictConfig,
    train_client_count: int,
    holdout_client_count: int,
    logger: Optional[ServerAgentFileLogger] = None,
) -> None:
    if not _cfg_bool(config, "eval.enable_federated_eval", True):
        return
    cadence = int(_cfg_get(config, "eval.every", 1))
    scheme = str(_cfg_get(config, "eval.configs.scheme", "dataset")).strip().lower()
    if scheme == "client":
        holdout_ratio = float(_cfg_get(config, "eval.configs.client_ratio", 0.0))
        holdout_counts = int(_cfg_get(config, "eval.configs.client_counts", 0))
        msg = (
            "Federated eval policy: "
            f"interval={'final_only' if cadence <= 0 else cadence} "
            "scheme=client "
            f"basis_clients={int(holdout_client_count)} "
            f"(client_ratio={holdout_ratio:.4f}, client_counts={holdout_counts})"
        )
    else:
        msg = (
            "Federated eval policy: "
            f"interval={'final_only' if cadence <= 0 else cadence} "
            f"scheme={scheme} "
            f"basis_clients={int(train_client_count)}"
        )
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)

def _warn_if_workers_pinned_to_single_device(
    config: DictConfig,
    world_size: int,
    logger: Optional[ServerAgentFileLogger] = None,
) -> None:
    if world_size <= 1:
        return
    dev = str(_cfg_get(config, "experiment.device", "cpu")).strip().lower()
    if not dev.startswith("cuda:"):
        return
    suffix = dev.split(":", 1)[1].strip()
    if not suffix.isdigit():
        return
    msg = (
        f"Device warning: `device={dev}` pins all ranks to the same GPU index. "
        "For multi-rank GPU spreading, use `device=cuda`."
    )
    if logger is not None:
        logger.warning(msg)
    else:
        print(msg)


def _trainer_metric_title(metric_name: str) -> str:
    text = str(metric_name).strip()
    if text == "":
        return "Metric"
    return text[:1].upper() + text[1:]


def _trainer_metric_value(
    stats_obj: Optional[Dict[str, Any]],
    metric_name: str,
) -> float:
    if not isinstance(stats_obj, dict):
        return -1.0
    nested = stats_obj.get("metrics", {})
    if isinstance(nested, dict):
        value = nested.get(metric_name, None)
        if isinstance(value, (int, float)):
            return float(value)
    for key in (f"metric_{metric_name}", metric_name):
        value = stats_obj.get(key, None)
        if isinstance(value, (int, float)):
            return float(value)
    return -1.0


def _build_trainer_log_row(
    *,
    mode: str,
    has_any_eval_split: bool,
    has_val_split: bool,
    has_test_split: bool,
    metric_names_for_log: List[str],
    epoch_idx: Optional[int],
    pre_eval_flag: str,
    elapsed: Any,
    train_stats_obj: Optional[Dict[str, Any]],
    val_stats_obj: Optional[Dict[str, Any]],
    test_stats_obj: Optional[Dict[str, Any]],
) -> List[Any]:
    row: List[Any] = []
    if str(mode).strip().lower() == "epoch":
        row.append(epoch_idx if epoch_idx is not None else "-")
    if bool(has_any_eval_split):
        row.append(pre_eval_flag)
    row.append(elapsed)
    row.append(
        float(train_stats_obj["loss"])
        if isinstance(train_stats_obj, dict) and "loss" in train_stats_obj
        else "-"
    )
    for metric_name in metric_names_for_log:
        row.append(_trainer_metric_value(train_stats_obj, metric_name))
    if bool(has_val_split):
        row.append(
            float(val_stats_obj["loss"])
            if isinstance(val_stats_obj, dict) and "loss" in val_stats_obj
            else -1.0
        )
        for metric_name in metric_names_for_log:
            row.append(_trainer_metric_value(val_stats_obj, metric_name))
    if bool(has_test_split):
        row.append(
            float(test_stats_obj["loss"])
            if isinstance(test_stats_obj, dict) and "loss" in test_stats_obj
            else -1.0
        )
        for metric_name in metric_names_for_log:
            row.append(_trainer_metric_value(test_stats_obj, metric_name))
    return row


def _build_trainer_log_title(
    *,
    mode: str,
    has_any_eval_split: bool,
    has_val_split: bool,
    has_test_split: bool,
    metric_names_for_log: List[str],
) -> List[str]:
    title: List[str] = []
    if str(mode).strip().lower() == "epoch":
        title.append("Epoch")
    if bool(has_any_eval_split):
        title.append("Pre Eval?")
    title.extend(["Time", "Train. Loss"])
    for metric_name in metric_names_for_log:
        title.append(f"Train. {_trainer_metric_title(metric_name)}")
    if bool(has_val_split):
        title.append("Val. Loss")
        for metric_name in metric_names_for_log:
            title.append(f"Val. {_trainer_metric_title(metric_name)}")
    if bool(has_test_split):
        title.append("Test Loss")
        for metric_name in metric_names_for_log:
            title.append(f"Test {_trainer_metric_title(metric_name)}")
    return title

def _entity_line(title: str, body: str) -> str:
    return f"  {title:<18} {body}"

def _join_metric_parts(parts: List[str]) -> str:
    if not parts:
        return ""
    return " | ".join(f"{part:<24}" for part in parts).rstrip()

def _pick_numeric(d: Dict[str, Any], candidates: List[str]) -> Optional[Tuple[str, float]]:
    for key in candidates:
        if key in d and isinstance(d[key], (int, float)):
            return key, float(d[key])
    return None

def _collect_client_values(
    all_stats: Dict[int, Dict[str, Any]],
    candidates: List[str],
) -> List[float]:
    values: List[float] = []
    for row in all_stats.values():
        hit = _pick_numeric(row, candidates)
        if hit is None:
            continue
        _, value = hit
        values.append(float(value))
    return values

def _summarize_client_metric_block(
    stats: Dict[int, Dict[str, Any]],
    eval_metric_order: List[str],
    *,
    loss_candidates: List[str],
    metric_candidates_for: Callable[[str], List[str]],
) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    section: Dict[str, Dict[str, float]] = {}
    parts: List[str] = []
    items: List[Tuple[str, List[str]]] = [("loss", list(loss_candidates))]
    items.extend(
        (metric_name, metric_candidates_for(metric_name))
        for metric_name in eval_metric_order
    )
    for label, candidates in items:
        vals = _collect_client_values(stats, candidates)
        if not vals:
            continue
        avg_value = float(np.mean(vals))
        std_value = float(np.std(vals))
        section[label] = {"avg": avg_value, "std": std_value}
        parts.append(f"{label}: {avg_value:.4f}/{std_value:.4f}")
    return section, parts

def _summarize_eval_metric_block(
    metrics: Dict[str, Any],
    eval_metric_order: List[str],
    *,
    with_client_std: bool,
) -> Tuple[Dict[str, object], List[str]]:
    section_metrics: Dict[str, object] = {}
    parts: List[str] = []
    used_raw_keys: set[str] = set()
    items: List[Tuple[str, List[str]]] = [("loss", ["loss"])]
    items.extend(
        (metric_name, [f"metric_{metric_name}", metric_name])
        for metric_name in eval_metric_order
    )
    for label, candidates in items:
        hit = _pick_numeric(metrics, candidates)
        if hit is None:
            continue
        raw_key, value = hit
        if raw_key in used_raw_keys:
            continue
        used_raw_keys.add(raw_key)
        if with_client_std:
            std_key = f"{raw_key}_std"
            if std_key in metrics and isinstance(metrics[std_key], (int, float)):
                std_val = float(metrics[std_key])
                section_metrics[label] = {"avg": float(value), "std": std_val}
                parts.append(f"{label}: {float(value):.4f}/{std_val:.4f}")
                continue
        section_metrics[label] = float(value)
        parts.append(f"{label}: {float(value):.4f}")
    return section_metrics, parts

def _federated_extrema_payload(
    metrics: Dict[str, Any],
    eval_metric_order: List[str],
) -> Dict[str, Dict[str, float]]:
    extrema: Dict[str, Dict[str, float]] = {}
    raw = {}
    for key, value in metrics.items():
        if (
            not key.endswith("_min")
            or not isinstance(value, (int, float))
            or f"{key[:-4]}_max" not in metrics
            or not isinstance(metrics[f"{key[:-4]}_max"], (int, float))
        ):
            continue
        base = key[:-4]
        raw[base] = {
            "label": base[7:] if base.startswith("metric_") else base,
            "min": float(value),
            "max": float(metrics[f"{base}_max"]),
        }
    if not raw:
        return extrema
    ordered_keys: List[str] = []
    if "loss" in raw:
        ordered_keys.append("loss")
    for metric_name in eval_metric_order:
        for candidate in (f"metric_{metric_name}", metric_name):
            if candidate in raw and candidate not in ordered_keys:
                ordered_keys.append(candidate)
                break
    if not ordered_keys:
        ordered_keys = sorted(raw.keys())
    for key in ordered_keys:
        item = raw[key]
        extrema[str(item["label"])] = {
            "min": float(item["min"]),
            "max": float(item["max"]),
        }
    return extrema

def _build_round_metrics_payload(
    config: DictConfig,
    selected_count: int,
    total_train_clients: int,
    stats,
    round_local_steps: Optional[int],
    round_wall_time_sec: Optional[float],
    global_gen_error: Optional[float],
    global_eval_metrics: Optional[Dict[str, float]],
    federated_eval_metrics: Optional[Dict[str, float]],
    federated_eval_in_metrics: Optional[Dict[str, float]],
    federated_eval_out_metrics: Optional[Dict[str, float]],
    track_gen_rewards: bool,
    round_gen_reward: Optional[float],
    cumulative_gen_reward: Optional[float],
) -> _RoundMetricsPayload:
    eval_metric_order = parse_metric_names(_cfg_get(config, "eval.metrics", None))
    if not eval_metric_order:
        eval_metric_order = ["acc1"]

    round_metrics: _RoundMetricsPayload = {
        "clients": {
            "selected": int(selected_count),
            "total": int(total_train_clients),
        }
    }
    if round_local_steps is not None:
        round_metrics["policy"] = {"tau_t": int(round_local_steps)}
    if isinstance(round_wall_time_sec, (int, float)):
        round_metrics["timing"] = {"round_wall_time_sec": float(round_wall_time_sec)}

    if stats:
        training_metrics, train_parts = _summarize_client_metric_block(
            stats,
            eval_metric_order,
            loss_candidates=["loss"],
            metric_candidates_for=lambda name: [f"metric_{name}", name],
        )
        if train_parts:
            round_metrics["training"] = training_metrics

        def _maybe_add_local_eval(json_key: str, prefix: str, enabled: bool) -> None:
            if not enabled:
                return
            section, parts = _summarize_client_metric_block(
                stats,
                eval_metric_order,
                loss_candidates=[f"{prefix}loss"],
                metric_candidates_for=lambda name: [f"{prefix}metric_{name}", f"{prefix}{name}"],
            )
            if parts:
                round_metrics[json_key] = section

        do_pre_evaluation = _cfg_bool(config, "eval.do_pre_evaluation", True)
        do_post_evaluation = _cfg_bool(config, "eval.do_post_evaluation", True)
        _maybe_add_local_eval("local_pre_val", "pre_val_", do_pre_evaluation)
        _maybe_add_local_eval("local_post_val", "post_val_", do_post_evaluation)
        _maybe_add_local_eval("local_pre_test", "pre_test_", do_pre_evaluation)
        _maybe_add_local_eval("local_post_test", "post_test_", do_post_evaluation)

        vals = _collect_client_values(stats, ["local_gen_error"])
        if vals:
            round_metrics["local_gen_error"] = {
                "avg": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }

    if isinstance(global_gen_error, (int, float)):
        round_metrics["global_gen_error"] = float(global_gen_error)

    if track_gen_rewards:
        round_metrics["gen_reward"] = {
            "round": float(round_gen_reward)
            if isinstance(round_gen_reward, (int, float))
            else None,
            "cumulative": float(cumulative_gen_reward)
            if isinstance(cumulative_gen_reward, (int, float))
            else None,
        }

    def _maybe_add_eval_block(
        json_key: str,
        metrics: Optional[Dict[str, float]],
        with_client_std: bool,
    ) -> None:
        if not isinstance(metrics, dict):
            return
        section_metrics, parts = _summarize_eval_metric_block(
            metrics,
            eval_metric_order,
            with_client_std=with_client_std,
        )
        if parts:
            round_metrics[json_key] = section_metrics

    _maybe_add_eval_block("global_eval", global_eval_metrics, with_client_std=False)
    _maybe_add_eval_block("fed_eval", federated_eval_metrics, with_client_std=True)
    _maybe_add_eval_block("fed_eval_in", federated_eval_in_metrics, with_client_std=True)
    _maybe_add_eval_block("fed_eval_out", federated_eval_out_metrics, with_client_std=True)

    extrema_source = (
        federated_eval_metrics
        if isinstance(federated_eval_metrics, dict)
        else federated_eval_in_metrics
        if isinstance(federated_eval_in_metrics, dict)
        else None
    )
    if isinstance(extrema_source, dict):
        extrema = _federated_extrema_payload(extrema_source, eval_metric_order)
        if extrema:
            round_metrics["fed_extrema"] = extrema
    return round_metrics

def _render_round_summary_lines(round_metrics: _RoundMetricsPayload) -> List[str]:
    clients = round_metrics.get("clients", {})
    selected = int(clients.get("selected", 0)) if isinstance(clients, dict) else 0
    total = int(clients.get("total", 0)) if isinstance(clients, dict) else 0
    pct = (100.0 * float(selected) / float(max(1, total)))
    lines = [
        "--- Round Summary ---",
        _entity_line("Clients:", f"selected={selected}/{total} ({pct:.2f}%)"),
    ]

    policy = round_metrics.get("policy", {})
    if isinstance(policy, dict) and "tau_t" in policy:
        lines.append(_entity_line("Policy:", f"tau_t={int(policy['tau_t'])}"))
    timing = round_metrics.get("timing", {})
    if isinstance(timing, dict) and isinstance(timing.get("round_wall_time_sec"), (int, float)):
        lines.append(
            _entity_line("Round Time:", f"{float(timing['round_wall_time_sec']):.3f}s")
        )

    def _section_parts(section: object) -> List[str]:
        if not isinstance(section, dict):
            return []
        parts: List[str] = []
        for label, value in section.items():
            if isinstance(value, dict) and all(k in value for k in ("avg", "std")):
                parts.append(f"{label}: {float(value['avg']):.4f}/{float(value['std']):.4f}")
            elif isinstance(value, (int, float)):
                parts.append(f"{label}: {float(value):.4f}")
        return parts

    train_local_specs = [
        ("training", "Training:"),
        ("local_pre_val", "Local Pre-val.:"),
        ("local_post_val", "Local Post-val.:"),
        ("local_pre_test", "Local Pre-test.:"),
        ("local_post_test", "Local Post-test.:"),
    ]
    for key, title in train_local_specs:
        parts = _section_parts(round_metrics.get(key))
        if parts:
            lines.append(_entity_line(title, _join_metric_parts(parts)))

    local_gen_error = round_metrics.get("local_gen_error", {})
    if (
        isinstance(local_gen_error, dict)
        and isinstance(local_gen_error.get("avg"), (int, float))
        and isinstance(local_gen_error.get("std"), (int, float))
    ):
        lines.append(
            _entity_line(
                "Local Gen. Error:",
                _join_metric_parts(
                    [
                        f"err.: {float(local_gen_error['avg']):.4f}/{float(local_gen_error['std']):.4f}"
                    ]
                ),
            )
        )
    if isinstance(round_metrics.get("global_gen_error"), (int, float)):
        lines.append(
            _entity_line(
                "Global Gen. Error:",
                _join_metric_parts([f"err.: {float(round_metrics['global_gen_error']):.4f}"]),
            )
        )
    gen_reward = round_metrics.get("gen_reward", {})
    if isinstance(gen_reward, dict):
        round_text = (
            f"{float(gen_reward['round']):.4f}"
            if isinstance(gen_reward.get("round"), (int, float))
            else "n/a"
        )
        cum_text = (
            f"{float(gen_reward['cumulative']):.4f}"
            if isinstance(gen_reward.get("cumulative"), (int, float))
            else "n/a"
        )
        lines.append(
            _entity_line(
                "Gen. Reward:",
                _join_metric_parts([f"round: {round_text}", f"cumulative: {cum_text}"]),
            )
        )

    eval_specs = [
        ("global_eval", "Global Eval.:"),
        ("fed_eval", "Federated Eval.:"),
        ("fed_eval_in", "Federated Eval(In).:"),
        ("fed_eval_out", "Federated Eval(Out).:"),
    ]
    for key, title in eval_specs:
        parts = _section_parts(round_metrics.get(key))
        if parts:
            lines.append(_entity_line(title, _join_metric_parts(parts)))

    fed_extrema = round_metrics.get("fed_extrema", {})
    if isinstance(fed_extrema, dict) and fed_extrema:
        items = [(k, v) for k, v in fed_extrema.items() if isinstance(v, dict)]
        shown = items[:4]
        parts = [
            f"{label}[min,max]=[{float(val.get('min', -1.0)):.4f},{float(val.get('max', -1.0)):.4f}]"
            for label, val in shown
        ]
        if len(items) > len(shown):
            parts.append(f"...(+{len(items) - len(shown)} more)")
        lines.append(_entity_line("Federated Extrema:", _join_metric_parts(parts)))
    return lines

def _log_round(
    config: DictConfig,
    round_idx: int,
    selected_count: int,
    total_train_clients: int,
    stats,
    round_local_steps: Optional[int] = None,
    round_wall_time_sec: Optional[float] = None,
    global_gen_error: Optional[float] = None,
    global_eval_metrics: Optional[Dict[str, float]] = None,
    federated_eval_metrics: Optional[Dict[str, float]] = None,
    federated_eval_in_metrics: Optional[Dict[str, float]] = None,
    federated_eval_out_metrics: Optional[Dict[str, float]] = None,
    track_gen_rewards: bool = False,
    round_gen_reward: Optional[float] = None,
    cumulative_gen_reward: Optional[float] = None,
    logger: ServerAgentFileLogger | None = None,
    tracker=None,
):
    round_metrics = _build_round_metrics_payload(
        config=config,
        selected_count=selected_count,
        total_train_clients=total_train_clients,
        stats=stats,
        round_local_steps=round_local_steps,
        round_wall_time_sec=round_wall_time_sec,
        global_gen_error=global_gen_error,
        global_eval_metrics=global_eval_metrics,
        federated_eval_metrics=federated_eval_metrics,
        federated_eval_in_metrics=federated_eval_in_metrics,
        federated_eval_out_metrics=federated_eval_out_metrics,
        track_gen_rewards=track_gen_rewards,
        round_gen_reward=round_gen_reward,
        cumulative_gen_reward=cumulative_gen_reward,
    )
    lines = _render_round_summary_lines(round_metrics)
    log = "\n".join(lines)
    if logger is not None:
        logger.info(log, round_label=f"Round {round_idx:04d}")
    else:
        print(log)
    if tracker is not None:
        tracker.log_metrics(step=round_idx, metrics=round_metrics)

def _new_server_logger(config: DictConfig, mode: str, run_id: str) -> ServerAgentFileLogger:
    run_dir = _resolve_run_dir_path(config, run_id)
    mode_text = str(mode).strip().lower()
    file_name = "server.log"
    if "-rank" in mode_text:
        suffix = mode_text.split("-rank", 1)[1].strip("-_ ")
        if suffix.isdigit() and int(suffix) != 0:
            file_name = f"server-rank{suffix}.log"
    return ServerAgentFileLogger(
        file_dir=str(run_dir),
        file_name=file_name,
        experiment_id=str(_cfg_get(config, "experiment.name", "appfl-sim")),
    )

def _resolve_run_log_dir(config: DictConfig, run_id: str) -> str:
    return str(_resolve_run_dir_path(config, run_id))

def _resolve_run_id() -> str:
    return f"{datetime.now().strftime('%y%m%d%H%M%S')}_{os.getpid()}"

def _start_summary_lines(
    mode: str,
    config: DictConfig,
    num_clients: int,
    train_client_count: int,
    holdout_client_count: int,
    num_sampled_clients: int,
) -> str:
    sampled_pct = (
        100.0 * float(num_sampled_clients) / float(max(1, train_client_count))
    )
    eval_scheme = str(_cfg_get(config, "eval.configs.scheme", "dataset")).strip().lower()
    lines = [
        f"Start {mode.upper()} simulation",
        f"  * Experiment: {_cfg_get(config, 'experiment.name', 'appfl-sim')}",
        f"  * Algorithm: {_cfg_get(config, 'algorithm.name', 'fedavg')}",
        f"  * Model: {_cfg_get(config, 'model.name', 'SimpleCNN')}",
        f"  * Dataset: {_cfg_get(config, 'dataset.name', 'MNIST')}",
        f"  * Rounds: {_cfg_get(config, 'train.num_rounds', 20)}",
        f"  * Total Clients: {num_clients}",
        f"  * Sampled Clients/Round: {num_sampled_clients}/{train_client_count} ({sampled_pct:.2f}%)",
        f"  * Evaluation Scheme: {eval_scheme}",
    ]
    if eval_scheme == "client":
        lines.append(f"  * Holdout Clients (evaluation): {holdout_client_count}")
    return "\n".join(lines)
