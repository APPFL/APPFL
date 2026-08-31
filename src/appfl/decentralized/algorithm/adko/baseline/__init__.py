"""Contextual baselines -- how an agent decides what counts as SUCCESS.

Built-ins are re-exported here; user-defined baselines load from a file path, following the
same convention APPFL uses for custom aggregators and trainers::

    # a built-in, by name
    baseline = get_appfl_baseline("RunningMedianBaseline")

    # your own, from any file
    baseline = get_appfl_baseline(
        "QuantileBaseline",
        baseline_path="./my_baselines.py",
        baseline_kwargs={"quantile": 0.9},
    )

See :class:`BaseBaseline` for the two-method contract and a worked custom example.
"""

from typing import Any, Dict, Optional

from appfl.decentralized.algorithm.adko.baseline.base_baseline import BaseBaseline
from appfl.decentralized.algorithm.adko.baseline.fixed_baseline import FixedBaseline
from appfl.decentralized.algorithm.adko.baseline.running_median_baseline import (
    RunningMedianBaseline,
    median,
    standard_deviation,
)

__all__ = [
    "BaseBaseline",
    "FixedBaseline",
    "RunningMedianBaseline",
    "median",
    "standard_deviation",
    "get_appfl_baseline",
    "build_baseline",
]

# Short aliases so a YAML config can say `baseline: running_median` rather than spelling the
# class name. Both forms resolve; the class name is canonical.
_ALIASES = {
    "fixed": "FixedBaseline",
    "running_median": "RunningMedianBaseline",
}


def get_appfl_baseline(
    baseline_name: str,
    baseline_kwargs: Optional[Dict[str, Any]] = None,
    baseline_path: Optional[str] = None,
) -> BaseBaseline:
    """Resolve a baseline by name, or load a user-defined one from ``baseline_path``.

    Mirrors ``get_appfl_aggregator`` / ``create_instance_from_file`` in ``appfl.misc.utils``,
    so a custom baseline is configured exactly like a custom aggregator: name the class, point
    at the file, pass its kwargs.

    :param baseline_name: a built-in class name (``"FixedBaseline"``), a short alias
        (``"running_median"``), or the class name to look for inside ``baseline_path``.
    :param baseline_kwargs: constructor arguments.
    :param baseline_path: path to a Python file defining the class. When given, the built-in
        registry is bypassed entirely.
    """
    kwargs = dict(baseline_kwargs or {})

    if baseline_path is not None:
        from appfl.misc.utils import create_instance_from_file

        instance = create_instance_from_file(baseline_path, baseline_name, **kwargs)
        if not isinstance(instance, BaseBaseline):
            # Not fatal -- duck typing would work -- but almost always a mistake worth
            # naming, since the alternative is a confusing AttributeError several rounds in.
            raise TypeError(
                f"{baseline_name!r} from {baseline_path} does not subclass BaseBaseline; "
                f"it must implement observe() and current()."
            )
        return instance

    canonical = _ALIASES.get(baseline_name, baseline_name)
    import importlib

    module = importlib.import_module("appfl.decentralized.algorithm.adko.baseline")
    try:
        BaselineClass = getattr(module, canonical)
    except AttributeError:
        available = sorted(set(_ALIASES) | {n for n in __all__ if n.endswith("Baseline")})
        raise ValueError(
            f"Invalid baseline name: {baseline_name!r}. Built-ins: {available}. "
            f"For a custom baseline, pass baseline_path to load it from a file."
        ) from None
    return BaselineClass(**kwargs)


def build_baseline(kind: str, **kwargs) -> BaseBaseline:
    """Convenience wrapper: ``build_baseline("fixed", threshold=50, scale=50)``.

    Equivalent to :func:`get_appfl_baseline` with kwargs inlined. Kept because it reads
    better at a call site that is constructing one baseline with literal arguments.
    """
    return get_appfl_baseline(kind, baseline_kwargs=kwargs)
