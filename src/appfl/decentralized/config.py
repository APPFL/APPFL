"""Loading the things a config names: callables, keyword blocks, agent ids.

What a decentralized run needs that OmegaConf does not give for free. The federation-wide
loaders that used to live here are gone: on this branch a node's config is complete on its
own, so there is nothing to merge a shared file into.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any, Callable

from omegaconf import Container, DictConfig, OmegaConf


def as_kwargs(node: Any) -> dict[str, Any]:
    """A plain dict from a config node that may be absent, empty, or already a dict.

    Needed because an empty ``DictConfig`` is falsy, so the usual ``node or {}`` turns it into
    something :func:`OmegaConf.to_container` refuses.
    """
    if node is None:
        return {}
    if isinstance(node, Container):
        return OmegaConf.to_container(node, resolve=True) or {}
    return dict(node)


def load_callable(path: str | Path, name: str) -> Callable[..., Any]:
    """Import ``name`` from the file at ``path``.

    Equivalent to :func:`appfl.misc.get_function_from_file`, except that an import error
    propagates instead of being printed and turned into ``None`` -- which matters here because
    the usual cause is a missing optional dependency (BoTorch, Olympus) and the traceback is
    the answer.
    """
    file_path = Path(path).resolve()
    module_dir = str(file_path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
    if spec is None or spec.loader is None:  # pragma: no cover -- unreadable file
        raise ImportError(f"cannot import {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, name):
        raise AttributeError(f"{file_path} defines no {name!r}")
    return getattr(module, name)


def accept_context(evaluator: Callable[..., float]) -> Callable[..., float]:
    """An evaluator that may be called as ``evaluator(point, **context)``.

    The agent passes its evaluation context (``agent_id``, ``round_idx``, ``eval_idx``) as
    keywords. An evaluator that takes ``**kwargs`` receives all of it as is. Otherwise it
    receives the context keys it names as parameters -- ``f(point, eval_idx)`` gets
    ``eval_idx`` -- and a plain ``f(point)``, which most oracles are, gets only the point.
    """
    try:
        parameters = list(inspect.signature(evaluator).parameters.values())
    except (TypeError, ValueError):  # pragma: no cover -- builtins without a signature
        parameters = []
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters):
        return evaluator
    # Everything after the point that can be passed by keyword.
    named = {
        p.name
        for p in parameters[1:]
        if p.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    if not named:
        return lambda point, **context: evaluator(point)
    return lambda point, **context: evaluator(
        point, **{key: value for key, value in context.items() if key in named}
    )


def agent_ids(federation_config: DictConfig) -> list[str]:
    """``agent-0 ... agent-N``, unless the config names them explicitly.

    Explicit names matter in a real deployment, where an agent id is a site rather than an
    index, and the relay's topology has to use the same ones.
    """
    federation = federation_config.federation_configs
    named = federation.get("agent_ids", None)
    if named:
        return [str(a) for a in named]
    return [f"agent-{i}" for i in range(int(federation.num_agents))]
