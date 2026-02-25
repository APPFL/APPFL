"""
appfl.misc.config_utils: shared config navigation and component builder utilities.

Ported from appfl.sim.misc.config_utils for use across the full APPFL package.
"""
from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Optional

import torch
from omegaconf import DictConfig, OmegaConf


# ---------------------------------------------------------------------------
# Config navigation
# ---------------------------------------------------------------------------

def _cfg_to_dict(cfg) -> Dict:
    if isinstance(cfg, DictConfig):
        return dict(OmegaConf.to_container(cfg, resolve=True))
    if isinstance(cfg, SimpleNamespace):
        return dict(vars(cfg))
    if isinstance(cfg, dict):
        return dict(cfg)
    if hasattr(cfg, "__dict__"):
        return dict(vars(cfg))
    return {}


def _cfg_get(config: DictConfig | dict, path: str, default: Any = None) -> Any:
    parts = [p for p in str(path).split(".") if p]
    current: Any = config
    for part in parts:
        if isinstance(current, DictConfig):
            if part not in current:
                return default
            current = current.get(part)
            continue
        if isinstance(current, dict):
            if part not in current:
                return default
            current = current.get(part)
            continue
        return default
    return default if current is None else current


def _cfg_set(config: DictConfig | dict, path: str, value: Any) -> None:
    parts = [p for p in str(path).split(".") if p]
    if not parts:
        return
    current: Any = config
    for part in parts[:-1]:
        if isinstance(current, DictConfig):
            if part not in current or current.get(part) is None:
                current[part] = OmegaConf.create({})
            current = current.get(part)
            continue
        if isinstance(current, dict):
            if part not in current or current.get(part) is None:
                current[part] = {}
            current = current.get(part)
            continue
        return
    last = parts[-1]
    if isinstance(current, DictConfig):
        current[last] = value
    elif isinstance(current, dict):
        current[last] = value


# ---------------------------------------------------------------------------
# Dynamic component loading
# ---------------------------------------------------------------------------

def _get_last_class_name_from_file(file_path: str) -> Optional[str]:
    with open(file_path) as f:
        tree = ast.parse(f.read())
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    return classes[-1].name if classes else None


def _get_last_function_name_from_file(file_path: str) -> Optional[str]:
    with open(file_path) as f:
        tree = ast.parse(f.read())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    return functions[-1].name if functions else None


def _load_module_from_file(file_path: str):
    file_path = os.path.abspath(file_path)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    module_dir, module_file = os.path.split(file_path)
    module_name, _ = os.path.splitext(module_file)
    if module_dir not in sys.path:
        sys.path.append(module_dir)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load python module from: {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_named_symbol(path: str, name: str):
    file_path = Path(str(path)).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"Custom component file not found: {file_path}")
    module_name = f"_appfl_custom_{file_path.stem}_{abs(hash(str(file_path)))}"
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import custom component from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if str(name).strip() == "":
        exported = [
            obj for obj in module.__dict__.values()
            if (inspect.isclass(obj) or inspect.isfunction(obj))
            and getattr(obj, "__module__", "") == module.__name__
        ]
        if not exported:
            raise ValueError(f"No class/function found in custom component file: {file_path}")
        return exported[-1]
    if not hasattr(module, name):
        raise ValueError(f"Custom component `{name}` not found in {file_path}")
    return getattr(module, name)


def _resolve_component_backend(kind: str, backend: str, path: str) -> str:
    mode = str(backend or "auto").strip().lower()
    if mode == "auto":
        return "custom" if str(path or "").strip() else "torch"
    if mode not in {"torch", "custom"}:
        raise ValueError(f"{kind}.backend must be one of: auto, torch, custom")
    return mode


# ---------------------------------------------------------------------------
# Loss and optimizer builders
# ---------------------------------------------------------------------------

def build_loss_from_config(config: DictConfig | dict):
    """Build a loss function from a config dict with a `loss` key."""
    name = str(_cfg_get(config, "loss.name", "CrossEntropyLoss"))
    backend = _resolve_component_backend(
        kind="loss",
        backend=str(_cfg_get(config, "loss.backend", "auto")),
        path=str(_cfg_get(config, "loss.path", "")),
    )
    kwargs = _cfg_to_dict(_cfg_get(config, "loss.configs", {}))
    if backend == "torch":
        if not hasattr(torch.nn, name):
            raise ValueError(f"Loss {name} not found in torch.nn")
        return getattr(torch.nn, name)(**kwargs)
    target = _load_named_symbol(str(_cfg_get(config, "loss.path", "")), name)
    if inspect.isclass(target):
        return target(**kwargs)
    if callable(target):
        return target(**kwargs)
    raise TypeError("Custom loss target must be a class or callable")


def build_loss_from_train_cfg(train_cfg: DictConfig | dict):
    """Build a loss function from a training config sub-dict."""
    payload = {
        "loss": {
            "name": _cfg_get(train_cfg, "loss.name", "CrossEntropyLoss"),
            "backend": _cfg_get(train_cfg, "loss.backend", "auto"),
            "path": _cfg_get(train_cfg, "loss.path", ""),
            "configs": _cfg_get(train_cfg, "loss.configs", {}),
        }
    }
    return build_loss_from_config(payload)


def build_optimizer_from_train_cfg(train_cfg: DictConfig | dict, params: Iterable):
    """Build a torch optimizer from a training config sub-dict."""
    name = str(_cfg_get(train_cfg, "optimizer.name", "SGD"))
    path = str(_cfg_get(train_cfg, "optimizer.path", ""))
    backend = _resolve_component_backend(
        kind="optimizer",
        backend=str(_cfg_get(train_cfg, "optimizer.backend", "auto")),
        path=path,
    )
    kwargs = _cfg_to_dict(_cfg_get(train_cfg, "optimizer.configs", {}))
    kwargs.setdefault("lr", float(_cfg_get(train_cfg, "optimizer.lr", 0.01)))
    if "weight_decay" not in kwargs:
        kwargs["weight_decay"] = float(
            _cfg_get(train_cfg, "optimizer.configs.weight_decay", 0.0)
        )
    if backend == "torch":
        if not hasattr(torch.optim, name):
            raise ValueError(f"Optimizer {name} not found in torch.optim")
        return getattr(torch.optim, name)(params, **kwargs)
    target = _load_named_symbol(path, name)
    if inspect.isclass(target):
        return target(params, **kwargs)
    if callable(target):
        return target(params, **kwargs)
    raise TypeError("Custom optimizer target must be a class or callable")
