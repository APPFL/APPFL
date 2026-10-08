"""Loading the things a config names: callables, paths, keyword blocks.

What a decentralized run needs that OmegaConf does not give for free. The federation-wide
loaders that used to live here are gone: on this branch a node's config is complete on its
own, so there is nothing to merge a shared file into.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from omegaconf import Container, DictConfig, OmegaConf


def as_kwargs(node: Any) -> Dict[str, Any]:
    """A plain dict from a config node that may be absent, empty, or already a dict.

    Needed because an empty ``DictConfig`` is falsy, so the usual ``node or {}`` turns it into
    something :func:`OmegaConf.to_container` refuses.
    """
    if node is None:
        return {}
    if isinstance(node, Container):
        return OmegaConf.to_container(node, resolve=True) or {}
    return dict(node)


def resolve_path(path: Union[str, Path], base_dir: Optional[Union[str, Path]] = None) -> Path:
    """Resolve a config-supplied path against the working directory, then ``base_dir``.

    APPFL's FL examples require you to run from ``examples/``; accepting a second origin lets
    a launcher be invoked from anywhere, which matters once launchers live in subdirectories.
    """
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    if base_dir is not None:
        fallback = Path(base_dir) / path
        if fallback.exists():
            return fallback
        raise FileNotFoundError(f"{path} (also tried {fallback})")
    raise FileNotFoundError(str(path))


def load_callable(path: Union[str, Path], name: str) -> Callable[..., Any]:
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


def agent_ids(federation_config: DictConfig) -> List[str]:
    """``agent-0 ... agent-N``, unless the config names them explicitly.

    Explicit names matter in a real deployment, where an agent id is a site rather than an
    index, and the relay's topology has to use the same ones.
    """
    federation = federation_config.federation_configs
    named = federation.get("agent_ids", None)
    if named:
        return [str(a) for a in named]
    return [f"agent-{i}" for i in range(int(federation.num_agents))]
