from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence

import torch

from appfl.metrics.metricszoo import get_metric


def parse_metric_names(raw: Any) -> List[str]:
    """Normalize metric names from config/CLI formats."""
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        if text == "":
            return []
        if "," in text:
            return [name.strip().lower() for name in text.split(",") if name.strip()]
        return [text.lower()]
    if isinstance(raw, Sequence):
        out: List[str] = []
        for item in raw:
            if item is None:
                continue
            name = str(item).strip().lower()
            if name:
                out.append(name)
        return out
    return [str(raw).strip().lower()]


class MetricsManager:
    """Manage running metric collectors and aggregate results.

    Behavior is intentionally similar to AAggFF's MetricManager while returning
    additional fields useful for APPFL-SIM logging.
    """

    def __init__(
        self,
        eval_metrics: Iterable[str] | str | None,
    ):
        metric_names = parse_metric_names(eval_metrics)
        if not metric_names:
            metric_names = ["acc1"]

        unique_names: List[str] = []
        for name in metric_names:
            if name not in unique_names:
                unique_names.append(name)

        self.metric_names = list(unique_names)
        self.metric_funcs = {name: get_metric(name) for name in self.metric_names}
        self.figures: Dict[str, float] = defaultdict(float)
        self._results: Dict[Any, Dict[str, Any]] | Dict[str, Any] = {}
        self._tracked_examples = 0

        # If Youden's J is used, enable threshold optimization for compatible metrics.
        self._enable_youdenj()

    def _enable_youdenj(self) -> None:
        if "youdenj" in self.metric_funcs:
            for func in self.metric_funcs.values():
                if hasattr(func, "_use_youdenj"):
                    setattr(func, "_use_youdenj", True)

    def _reset_collectors(self) -> None:
        self.metric_funcs = {name: get_metric(name) for name in self.metric_names}
        self._enable_youdenj()

    @staticmethod
    def _to_tensor(value: Any) -> torch.Tensor:
        if torch.is_tensor(value):
            return value.detach().cpu()
        return torch.as_tensor(value)

    def track(self, loss: float, pred: Any, true: Any) -> None:
        pred_t = self._to_tensor(pred)
        true_t = self._to_tensor(true)
        if true_t.ndim == 0:
            batch_size = 1
        else:
            batch_size = int(true_t.shape[0])

        self.figures["loss"] += float(loss) * batch_size
        self._tracked_examples += batch_size

        for module in self.metric_funcs.values():
            module.collect(pred_t, true_t)

    def aggregate(
        self,
        total_len: int | None = None,
        curr_step: Any | None = None,
    ) -> Dict[str, Any]:
        num_examples = int(self._tracked_examples if total_len is None else total_len)
        if num_examples <= 0 or self._tracked_examples <= 0:
            running_metrics = {name: -1.0 for name in self.metric_funcs.keys()}
        else:
            running_metrics = {}
            for name, module in self.metric_funcs.items():
                try:
                    running_metrics[name] = float(module.summarize())
                except Exception:
                    running_metrics[name] = -1.0
        loss = (
            float(self.figures["loss"] / max(num_examples, 1))
            if num_examples > 0
            else -1.0
        )

        # Keep both nested metrics and flattened metric_* keys for easy logging.
        result: Dict[str, Any] = {
            "loss": loss,
            "num_examples": num_examples,
            "metrics": running_metrics,
        }
        for name, value in running_metrics.items():
            result[f"metric_{name}"] = float(value)

        if curr_step is None:
            self._results = result
        else:
            if (
                not isinstance(self._results, dict)
                or "loss" in self._results
            ):
                self._results = {}
            self._results[curr_step] = result

        # Reset running states after aggregation.
        self.figures = defaultdict(float)
        self._tracked_examples = 0
        self._reset_collectors()
        return result

    @property
    def results(self):
        return self._results
