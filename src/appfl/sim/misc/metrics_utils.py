from __future__ import annotations
from typing import Any


def _weighted_mean(stats: dict[int, dict], key: str) -> float:
    total = 0.0
    count = 0
    for values in stats.values():
        if not isinstance(values, dict):
            continue
        if key not in values or not isinstance(values.get(key), (int, float)):
            continue
        n = int(values.get("num_examples", 0))
        total += float(values.get(key, 0.0)) * n
        count += n
    return total / count if count > 0 else 0.0


def _attach_prefixed_metrics(
    output: dict[str, Any],
    metrics: dict[str, Any] | None,
    prefix: str,
) -> None:
    if not isinstance(metrics, dict) or not metrics:
        return
    numeric_metrics = {
        str(key): float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float))
    }
    if not numeric_metrics:
        return
    output[f"{prefix}_metrics"] = numeric_metrics
    for key, value in numeric_metrics.items():
        output[f"{prefix}_metric_{key}"] = float(value)
