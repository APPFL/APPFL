import hashlib
import logging
import re
from typing import Dict


class LevelFilter(logging.Filter):
    def __init__(self, level):
        self.level = level

    def filter(self, record):
        return record.levelno == self.level


class _RoundAwareFormatter(logging.Formatter):
    """Formatter that injects an optional round label into log lines."""

    def __init__(self, pattern: str):
        super().__init__(pattern)

    def format(self, record: logging.LogRecord) -> str:
        round_label = getattr(record, "round_label", "")
        record.round_part = f" ({round_label})" if round_label else ""
        return super().format(record)


def _sanitize_wandb_token(token: str) -> str:
    text = re.sub(r"[^0-9a-zA-Z_]+", "_", str(token).strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return text.lower()


def _format_wandb_panel_key(panel: str, parts: list) -> str:
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


def _remap_server_wandb_payload(flat_payload: Dict[str, float]) -> Dict[str, float]:
    """Remap flat metric keys into structured WandB panel paths."""
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
        if (
            len(parts) == 2
            and parts[0] == "clients"
            and parts[1] in {"selected", "total"}
        ):
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
