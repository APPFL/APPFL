"""
appfl.metrics: FL evaluation metrics for APPFL.

Provides a structured metrics collection system including:
- BaseMetric: abstract base class for all metrics
- MetricsManager: accumulates batch-level results and aggregates per round
- 17 built-in metrics via METRIC_REGISTRY (acc1, acc5, auroc, f1, mse, etc.)
"""

from appfl.metrics.basemetric import BaseMetric
from appfl.metrics.manager import MetricsManager, parse_metric_names
from appfl.metrics.metricszoo import METRIC_REGISTRY, accuracy_from_logits, get_metric

__all__ = [
    "BaseMetric",
    "MetricsManager",
    "parse_metric_names",
    "METRIC_REGISTRY",
    "accuracy_from_logits",
    "get_metric",
]
