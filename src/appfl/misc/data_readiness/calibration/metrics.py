"""Fixed-bin calibration with local cell suppression."""

from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


_REPORT_KEYS = {
    "schema_version",
    "n_bins",
    "min_cell_count",
    "min_outcome_count",
    "cells",
}
_CELL_KEYS = {"n", "sum_p", "sum_y", "sum_sq_err"}


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _arrays(
    predictions: Sequence[float], outcomes: Sequence[int], groups: Sequence[Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        predictions = np.asarray(predictions, dtype=float)
        outcomes = np.asarray(outcomes, dtype=float)
        groups = np.asarray(groups, dtype=object)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid evaluation arrays") from error
    if any(array.ndim != 1 for array in (predictions, outcomes, groups)):
        raise ValueError("predictions, outcomes, and groups must be one-dimensional")
    if not len(predictions) == len(outcomes) == len(groups):
        raise ValueError("predictions, outcomes, and groups must have equal length")
    if not np.all(np.isfinite(predictions)) or np.any(
        (predictions < 0) | (predictions > 1)
    ):
        raise ValueError("predictions must be finite probabilities in [0, 1]")
    if not np.all(np.isin(outcomes, [0, 1])):
        raise ValueError("outcomes must contain only 0 and 1")
    labels = []
    for value in groups.tolist():
        if not isinstance(value, (str, Real)) or (
            isinstance(value, Real) and not np.isfinite(value)
        ):
            raise ValueError("group labels must be strings or finite numbers")
        label = str(value).strip()
        if not label:
            raise ValueError("group labels cannot be empty")
        labels.append(label)
    return predictions, outcomes, np.asarray(labels, dtype=str)


def build_calibration_report(
    predictions: Sequence[float],
    outcomes: Sequence[int],
    groups: Sequence[Any],
    n_bins: int = 10,
    min_cell_count: int = 5,
    release_n_bins: Optional[int] = None,
    min_outcome_count: int = 0,
) -> Dict[str, Any]:
    """Build a JSON-safe report containing only retained subgroup/bin totals.

    ``release_n_bins`` must divide ``n_bins``. Choose the same settings for all
    clients before evaluating data. Merging happens before local suppression.
    ``min_outcome_count`` requires that many events and non-events per cell.
    """
    n_bins = _integer(n_bins, "n_bins")
    released = (
        n_bins if release_n_bins is None else _integer(release_n_bins, "release_n_bins")
    )
    if n_bins % released:
        raise ValueError("release_n_bins must divide n_bins")
    minimum = _integer(min_cell_count, "min_cell_count")
    outcome_minimum = _integer(min_outcome_count, "min_outcome_count", 0)
    p, y, labels = _arrays(predictions, outcomes, groups)
    edges = np.arange(n_bins + 1, dtype=float) / n_bins
    index = np.searchsorted(edges[1:-1], p, side="right") // (n_bins // released)
    names, group_index = np.unique(labels, return_inverse=True)
    flat_index = group_index * released + index
    shape = (len(names), released)
    length = len(names) * released
    counts = np.bincount(flat_index, minlength=length).reshape(shape)
    sums = [
        np.bincount(flat_index, weights=weights, minlength=length).reshape(shape)
        for weights in (p, y, (p - y) ** 2)
    ]
    cells = {}
    for number, name in enumerate(names):
        retained = {}
        for bin_index in np.flatnonzero(counts[number] >= minimum):
            n = int(counts[number, bin_index])
            events = int(sums[1][number, bin_index])
            if min(events, n - events) < outcome_minimum:
                continue
            retained[str(int(bin_index))] = {
                "n": n,
                "sum_p": float(sums[0][number, bin_index]),
                "sum_y": events,
                "sum_sq_err": float(sums[2][number, bin_index]),
            }
        if retained:
            cells[str(name)] = retained
    return {
        "schema_version": 1,
        "n_bins": released,
        "min_cell_count": minimum,
        "min_outcome_count": outcome_minimum,
        "cells": cells,
    }


def _validate_cell(cell: Any, index: int, policy: Tuple[int, int, int]) -> None:
    if not isinstance(cell, Mapping) or set(cell) != _CELL_KEYS:
        raise ValueError("invalid calibration cell")
    n_bins, minimum, outcome_minimum = policy
    n = _integer(cell["n"], "cell count")
    for name in ("sum_p", "sum_y", "sum_sq_err"):
        value = cell[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(value)
        ):
            raise ValueError("cell sums must be finite numbers")
        if not 0 <= value <= n + 1e-12 * n:
            raise ValueError("cell sums must lie between zero and the count")
    events = cell["sum_y"]
    if not float(events).is_integer() or events > n:
        raise ValueError("outcome counts must be integers between zero and the count")
    if n < minimum or min(events, n - events) < outcome_minimum:
        raise ValueError("received a cell below the configured thresholds")
    tolerance = 1e-12 * n
    if (
        not index * n / n_bins - tolerance
        <= cell["sum_p"]
        <= (index + 1) * n / n_bins + tolerance
    ):
        raise ValueError("prediction sum is outside its probability bin")


def aggregate_calibration_reports(
    reports: Mapping, min_n_warn: int = 50
) -> Dict[str, Any]:
    """Merge reports keyed by the client IDs supplied by the communication layer.

    Scores apply to retained patients. Missing groups are unavailable; the
    reports do not contain the full counts needed to calculate coverage.
    """
    if not isinstance(reports, Mapping) or not reports:
        raise ValueError("at least one client report is required")
    min_n_warn = _integer(min_n_warn, "min_n_warn")
    policy = None
    merged = {}
    for report in reports.values():
        if not isinstance(report, Mapping) or set(report) != _REPORT_KEYS:
            raise ValueError("invalid calibration report")
        if type(report["schema_version"]) is not int or report["schema_version"] != 1:
            raise ValueError("unsupported calibration report schema")
        current = (
            _integer(report["n_bins"], "n_bins"),
            _integer(report["min_cell_count"], "min_cell_count"),
            _integer(report["min_outcome_count"], "min_outcome_count", 0),
        )
        if policy is not None and current != policy:
            raise ValueError("clients must use matching bins and thresholds")
        policy = current
        if not isinstance(report["cells"], Mapping):
            raise ValueError("cells must be a mapping")
        for group, bins in report["cells"].items():
            if (
                not isinstance(group, str)
                or not group.strip()
                or group != group.strip()
            ):
                raise ValueError("report group labels must be non-empty strings")
            if not isinstance(bins, Mapping) or not bins:
                raise ValueError("reported groups must contain retained cells")
            for key, cell in bins.items():
                try:
                    index = int(key)
                except (ValueError, TypeError, OverflowError) as error:
                    raise ValueError("bin indices must be integer strings") from error
                if (
                    not isinstance(key, str)
                    or str(index) != key
                    or not 0 <= index < policy[0]
                ):
                    raise ValueError("invalid probability bin index")
                _validate_cell(cell, index, policy)
                total = merged.setdefault(group, {}).setdefault(
                    index, {"n": 0, "sum_p": 0.0, "sum_y": 0.0, "sum_sq_err": 0.0}
                )
                for name in _CELL_KEYS:
                    total[name] += cell[name]
    groups = {}
    for group, bins in merged.items():
        n = int(sum(cell["n"] for cell in bins.values()))
        groups[group] = {
            "n": n,
            "ece": float(
                sum(abs(cell["sum_p"] - cell["sum_y"]) for cell in bins.values()) / n
            ),
            "brier": float(sum(cell["sum_sq_err"] for cell in bins.values()) / n),
            "warning": f"only {n} retained patients; estimate may be unreliable"
            if n < min_n_warn
            else None,
        }
    return {
        "status": "ok" if groups else "unavailable",
        "scope": "retained_patients",
        "n_bins": policy[0],
        "min_cell_count": policy[1],
        "min_outcome_count": policy[2],
        "groups": groups,
    }
