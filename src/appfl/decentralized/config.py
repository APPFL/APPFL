"""Configuration loading for a decentralized federation -- the algorithm-agnostic half.

Everything here reads keys that any decentralized method would need: who is in the
federation, how they are connected, how much traffic a link may carry, and where to find the
files that define the local problem. Nothing here imports an algorithm, so a second method
reuses it unchanged; ADKO's own keys are read by
:mod:`appfl.decentralized.algorithm.adko.builder`.

Two files, mirroring the server/client split of APPFL's federated-learning configs:

* **federation config** -- what every agent must agree on: the graph, the round count, the
  algorithm's settings, the surrogate. The analogue of a ``server_*.yaml``. Two sites running
  different copies of this file are running different experiments.
* **agent config** -- what belongs to one site: its slice of the design space, its oracle, and
  how it reaches the relay. The analogue of a ``client_N.yaml``, and the only file that
  describes anything private.

This lives in the library rather than in ``examples/`` because under a real deployment every
site builds its own agent from its own checkout. If the schema and the construction logic were
example code, each site would carry a fork of them, and two forks that have drifted produce
exactly the silent disagreement between sites that ``run_serial``/``run_mpi``/``run_site``
exist to detect.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from omegaconf import Container, DictConfig, OmegaConf

from appfl.decentralized.budget import CommBudget
from appfl.decentralized.topology import Topology, build_topology

__all__ = [
    "as_kwargs",
    "resolve_path",
    "load_callable",
    "load_federation_config",
    "load_agent_configs",
    "agent_ids",
    "create_topology",
    "create_budget",
]


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


def _resolve_resource_paths(config: Any, base_dir: Optional[Union[str, Path]]) -> None:
    """Rewrite every ``*_path`` in a loaded config to an absolute path, in place.

    Done once at load time, where the base directory is known, so that everything downstream
    works with absolute paths and a config printed after loading names exactly the files that
    will be imported. The convention is the one the FL configs already use -- ``model_path``,
    ``loss_fn_path``, ``dataset_path`` -- so the rule is the suffix, not a list of keys.
    """
    if isinstance(config, DictConfig):
        for key, value in config.items():
            if isinstance(value, str) and str(key).endswith("_path"):
                config[key] = str(resolve_path(value, base_dir).resolve())
            else:
                _resolve_resource_paths(value, base_dir)
    elif OmegaConf.is_list(config):
        for item in config:
            _resolve_resource_paths(item, base_dir)


def load_federation_config(
    path: Union[str, Path], base_dir: Optional[Union[str, Path]] = None
) -> DictConfig:
    """Load the shared configuration and make its resource paths absolute."""
    config = OmegaConf.load(str(resolve_path(path, base_dir)))
    _resolve_resource_paths(config, base_dir)
    return config


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


def load_agent_configs(
    path: Union[str, Path],
    federation_config: DictConfig,
    base_dir: Optional[Union[str, Path]] = None,
) -> List[DictConfig]:
    """One config per agent, with the per-agent fields filled in by index.

    The same move APPFL's ``run_serial.py`` makes for FL clients: load one template, then set
    the identity and the data slice. An agent's own config file only ever describes *an*
    agent; which one it is, is the launcher's business.
    """
    ids = agent_ids(federation_config)
    seed = int(federation_config.federation_configs.get("seed", 0))
    config_path = resolve_path(path, base_dir)
    configs = []
    for index, agent_id in enumerate(ids):
        config = OmegaConf.load(str(config_path))
        _check_same_experiment(federation_config, config, config_path)
        _resolve_resource_paths(config, base_dir)
        config.agent_id = agent_id
        config.space_configs.space_kwargs.agent_index = index
        config.space_configs.space_kwargs.num_agents = len(ids)
        config.space_configs.space_kwargs.seed = seed
        configs.append(config)
    return configs


def _check_same_experiment(
    federation_config: DictConfig, agent_config: DictConfig, agent_path: Path
) -> None:
    """Refuse a federation config and an agent config that describe different experiments.

    The two files are passed as separate flags and default independently, so forgetting one
    of them pairs, say, the Suzuki federation with the toy agent. That combination is not a
    crash: it builds a categorical GP over a one-dimensional interval and reports numbers
    that look like a result. An optional ``experiment`` tag in both files turns it into an
    error at load time. Untagged configs are not checked, so a config written before this
    existed still loads.
    """
    expected = federation_config.get("experiment", None)
    found = agent_config.get("experiment", None)
    if expected is None or found is None or str(expected) == str(found):
        return
    raise ValueError(
        f"config mismatch: the federation config is for experiment {str(expected)!r} but "
        f"{agent_path} is for {str(found)!r}. Pass the matching --agent_config; these two "
        f"files describe one experiment together and default independently."
    )


def create_topology(federation_config: DictConfig) -> Topology:
    """The federation graph. Every process builds it identically, from the same config."""
    federation = federation_config.federation_configs
    name = str(federation.get("topology", "fully_connected"))
    topology = build_topology(
        name, agent_ids(federation_config), **as_kwargs(federation.get("topology_kwargs", None))
    )
    if topology.fiedler_value() <= 1e-9:
        raise ValueError(
            f"topology '{name}' is disconnected; a decentralized method assumes a connected "
            f"graph and its convergence guarantees do not apply"
        )
    return topology


def create_budget(federation_config: DictConfig) -> CommBudget:
    """The per-link traffic cap, from config. ``null`` means unlimited."""
    comm = as_kwargs(federation_config.get("comm_configs", None))
    budget = as_kwargs(comm.get("budget_configs", None))
    return CommBudget(
        bits_per_neighbor_per_round=budget.get("bits_per_neighbor_per_round", None)
    )
