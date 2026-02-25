from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import torch

from appfl.sim.models import MODEL_REGISTRY, get_model_class


@dataclass
class ModelSpec:
    source: str
    name: str
    num_classes: int
    in_channels: int
    input_shape: Tuple[int, ...]
    context: Dict[str, Any]
    model_kwargs: Dict[str, Any]
    hf_task: str
    hf_pretrained: bool
    hf_local_files_only: bool
    hf_trust_remote_code: bool
    hf_gradient_checkpointing: bool
    cache_dir: str
    hf_kwargs: Dict[str, Any]
    hf_config_overrides: Dict[str, Any]


def _path_get(cfg: Dict[str, Any], path: str, default: Any = None) -> Any:
    parts = [p for p in str(path).split(".") if p]
    cur: Any = cfg
    for part in parts:
        if isinstance(cur, dict):
            if part not in cur:
                return default
            cur = cur[part]
            continue
        return default
    return default if cur is None else cur


def _as_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _safe_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _safe_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "no", "n", "off"}:
            return False
        return default
    return bool(value)


def _default_model_context(
    cfg: Dict[str, Any], input_shape: Tuple[int, ...], num_classes: int
) -> Dict[str, Any]:
    channels = int(input_shape[0]) if len(input_shape) >= 1 else 1
    spatial = input_shape[1:] if len(input_shape) >= 2 else (1,)
    if len(spatial) == 0:
        spatial = (1,)

    inferred_resize = int(spatial[0])
    inferred_in_features = 1
    for dim in input_shape:
        inferred_in_features *= int(dim)

    model_configs = _as_dict(_path_get(cfg, "model.configs", {}))
    need_embedding_cfg = _safe_bool(
        _path_get(cfg, "need_embedding", model_configs.get("need_embedding", False)),
        False,
    )
    seq_len_cfg = _safe_int(
        model_configs.get("seq_len", _path_get(cfg, "seq_len", 128)),
        128,
    )
    num_embeddings_cfg = _safe_int(
        model_configs.get("num_embeddings", _path_get(cfg, "num_embeddings", 10000)),
        10000,
    )
    context = {
        "model_name": _path_get(cfg, "model.name", "SimpleCNN"),
        "num_classes": _safe_int(num_classes, 0),
        "in_channels": channels,
        "in_features": inferred_in_features,
        "resize": inferred_resize,
        "hidden_size": _safe_int(model_configs.get("hidden_size", 64), 64),
        "dropout": _safe_float(model_configs.get("dropout", 0.0), 0.0),
        "num_layers": _safe_int(model_configs.get("num_layers", 2), 2),
        "num_embeddings": num_embeddings_cfg,
        "embedding_size": _safe_int(model_configs.get("embedding_size", 128), 128),
        "seq_len": seq_len_cfg,
        "need_embedding": need_embedding_cfg,
        "is_seq2seq": _safe_bool(model_configs.get("is_seq2seq", False), False),
        "B": _safe_int(_path_get(cfg, "train.batch_size", 32), 32),
    }
    context.update(model_configs)
    return context


def _parse_model_spec(
    cfg: Dict[str, Any],
    input_shape: Tuple[int, ...],
    num_classes: int,
) -> ModelSpec:
    context = _default_model_context(cfg, input_shape=input_shape, num_classes=num_classes)
    source = str(_path_get(cfg, "model.backend", "auto")).lower()
    name = str(_path_get(cfg, "model.name", "SimpleCNN")).strip()
    model_configs = _as_dict(_path_get(cfg, "model.configs", {}))
    needs_embedding = _safe_bool(context.get("need_embedding", False), False)
    if (
        needs_embedding
        and source in {"auto", "custom"}
        and name.lower() == "simplecnn"
    ):
        # Default image model is incompatible with tokenized/text datasets.
        # Use a minimal local text model unless user explicitly selects one.
        name = "StackedLSTM"
        context["model_name"] = name
        model_configs.setdefault("is_seq2seq", False)
    if source in {"hf", "torchvision", "torchtext", "torchaudio"} and not name:
        raise ValueError(
            f"model.backend={source} requires model.name to be set to the exact backend name/card."
        )
    model_path = str(_path_get(cfg, "model.path", "./appfl.sim/models"))
    in_channels = int(model_configs.get("in_channels", context.get("in_channels", 1)))
    resolved_num_classes = int(
        model_configs.get("num_classes", context.get("num_classes", num_classes))
    )

    return ModelSpec(
        source=source,
        name=name,
        num_classes=resolved_num_classes,
        in_channels=in_channels,
        input_shape=tuple(input_shape),
        context=context,
        model_kwargs=model_configs,
        hf_task=str(model_configs.get("hf_task", "sequence_classification")).strip().lower(),
        hf_pretrained=_safe_bool(model_configs.get("pretrained", False), False),
        hf_local_files_only=_safe_bool(model_configs.get("hf_local_files_only", False), False),
        hf_trust_remote_code=_safe_bool(model_configs.get("hf_trust_remote_code", False), False),
        hf_gradient_checkpointing=_safe_bool(model_configs.get("hf_gradient_checkpointing", False), False),
        cache_dir=str(
            model_configs.get(
                "cache_dir",
                model_configs.get("hf_cache_dir", model_path),
            )
        ),
        hf_kwargs=_as_dict(model_configs.get("hf_kwargs", {})),
        hf_config_overrides=_as_dict(model_configs.get("hf_config_overrides", {})),
    )


def _load_appfl_model(spec: ModelSpec):
    if spec.name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(
            f"model.backend=custom requires an exact local model name. "
            f"Got '{spec.name}'. Available: {available}"
        )

    model_class = get_model_class(spec.name)

    context = dict(spec.context)
    context.update(spec.model_kwargs)
    context["model_name"] = spec.name
    context["num_classes"] = spec.num_classes
    context["in_channels"] = spec.in_channels

    signature = inspect.signature(model_class.__init__)
    model_args: Dict[str, Any] = {}
    missing = []
    for name, param in signature.parameters.items():
        if name == "self":
            continue
        if name in context:
            model_args[name] = context[name]
            continue
        if param.default is not inspect._empty:
            model_args[name] = param.default
            continue
        missing.append(name)

    if missing:
        raise ValueError(
            f"Missing required model args for {spec.name}: {missing}. "
            "Pass them through config/CLI."
        )
    return model_class(**model_args)


def _filtered_model_kwargs(spec: ModelSpec) -> Dict[str, Any]:
    kwargs = dict(spec.model_kwargs)
    for reserved_key in (
        "pretrained",
        "hf_task",
        "hf_local_files_only",
        "hf_trust_remote_code",
        "hf_gradient_checkpointing",
        "hf_kwargs",
        "hf_config_overrides",
        "cache_dir",
        "hf_cache_dir",
    ):
        kwargs.pop(reserved_key, None)
    return kwargs


class _HFAdapter(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, task: str) -> None:
        super().__init__()
        self.model = model
        self.task = task

    def forward(self, x):
        kwargs: Dict[str, Any] = {}

        if isinstance(x, dict):
            kwargs = x
        elif isinstance(x, (list, tuple)) and len(x) == 2 and all(torch.is_tensor(v) for v in x):
            kwargs = {
                "input_ids": x[0].long(),
                "attention_mask": x[1].long(),
            }
        elif torch.is_tensor(x):
            if self.task in {"sequence_classification", "token_classification", "causal_lm"}:
                if x.ndim >= 3 and x.shape[1] >= 2:
                    kwargs = {
                        "input_ids": x[:, 0].long(),
                        "attention_mask": x[:, 1].long(),
                    }
                else:
                    kwargs = {"input_ids": x.long()}
            elif self.task == "vision_classification":
                kwargs = {"pixel_values": x.float()}
            else:
                kwargs = {"input_ids": x.long()}
        else:
            raise TypeError(f"Unsupported HuggingFace input type: {type(x)}")

        outputs = self.model(**kwargs)
        if hasattr(outputs, "logits"):
            return outputs.logits
        if isinstance(outputs, tuple) and len(outputs) > 0:
            return outputs[0]
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        return outputs


def _resolve_hf_model_id(spec: ModelSpec) -> str:
    return spec.name


def _resolve_hf_task(spec: ModelSpec) -> str:
    task = str(spec.hf_task).strip().lower()
    if task:
        return task
    if len(spec.input_shape) >= 2:
        return "vision_classification"
    return "sequence_classification"


def _build_hf_scratch_config(model_id: str, spec: ModelSpec, task: str):
    from transformers import AutoConfig

    overrides = dict(spec.hf_config_overrides)
    overrides.setdefault("num_labels", int(spec.num_classes))
    if task in {"sequence_classification", "token_classification", "causal_lm"}:
        overrides.setdefault("vocab_size", int(spec.context.get("num_embeddings", 10000)))
    try:
        return AutoConfig.from_pretrained(
            model_id,
            cache_dir=str(spec.cache_dir),
            local_files_only=bool(spec.hf_local_files_only),
            trust_remote_code=bool(spec.hf_trust_remote_code),
            **overrides,
        )
    except Exception as e:
        raise ValueError(
            f"pretrained=false requires accessible HF config for model '{model_id}'. "
            "Use `model.configs.hf_local_files_only=false` to allow download, "
            "or cache it first. If this model card is not a Transformers model "
            "(missing `config.json`), choose a Transformers-compatible model id."
        ) from e


def _load_hf_model(spec: ModelSpec):
    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForImageClassification,
            AutoModelForSequenceClassification,
            AutoModelForTokenClassification,
        )
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "huggingface backend requested but transformers is not installed. "
            "Install with: pip install transformers"
        ) from e

    task = _resolve_hf_task(spec)
    model_id = _resolve_hf_model_id(spec)
    pretrained = bool(spec.hf_pretrained)
    local_files_only = bool(spec.hf_local_files_only)
    trust_remote_code = bool(spec.hf_trust_remote_code)

    # Do not forward generic local-model config keys (e.g., hidden_size, num_layers)
    # to HF model constructors; only forward explicit hf_kwargs.
    common_kwargs: Dict[str, Any] = {}
    common_kwargs.update(spec.hf_kwargs)

    if task == "sequence_classification":
        model_cls = AutoModelForSequenceClassification
    elif task == "token_classification":
        model_cls = AutoModelForTokenClassification
    elif task == "causal_lm":
        model_cls = AutoModelForCausalLM
    elif task == "vision_classification":
        model_cls = AutoModelForImageClassification
    else:
        raise ValueError(
            "Unsupported hf task: "
            f"{task}. Use one of: sequence_classification, token_classification, causal_lm, vision_classification"
        )

    if pretrained:
        model = model_cls.from_pretrained(
            model_id,
            num_labels=int(spec.num_classes),
            cache_dir=str(spec.cache_dir),
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            **common_kwargs,
        )
    else:
        config = _build_hf_scratch_config(model_id, spec, task)
        model = model_cls.from_config(config)

    if bool(spec.hf_gradient_checkpointing) and hasattr(
        model, "gradient_checkpointing_enable"
    ):
        model.gradient_checkpointing_enable()

    return _HFAdapter(model=model, task=task)


def _load_torchvision_model(spec: ModelSpec):
    try:
        import torchvision.models as tv_models
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "torchvision backend requested but torchvision is not installed. "
            "Install with: pip install torchvision"
        ) from e

    kwargs = _filtered_model_kwargs(spec)
    pretrained = bool(spec.model_kwargs.get("pretrained", False))
    model_name = str(spec.name).strip()

    def _build_with_pruning(build_fn, init_kwargs: Dict[str, Any]):
        curr = dict(init_kwargs)
        while True:
            try:
                return build_fn(**curr)
            except TypeError as e:
                msg = str(e)
                hit = re.search(r"unexpected keyword argument ['\"]([^'\"]+)['\"]", msg)
                if hit is None:
                    raise
                bad_key = str(hit.group(1))
                if bad_key not in curr:
                    raise
                curr.pop(bad_key, None)

    if hasattr(tv_models, "get_model"):
        if pretrained:
            kwargs.setdefault("weights", "DEFAULT")
        kwargs.setdefault("num_classes", int(spec.num_classes))
        return _build_with_pruning(
            lambda **kw: tv_models.get_model(model_name, **kw),
            kwargs,
        )

    if not hasattr(tv_models, model_name):
        available = sorted(
            name
            for name in dir(tv_models)
            if callable(getattr(tv_models, name, None)) and not name.startswith("_")
        )
        raise ValueError(
            f"Unknown torchvision model '{model_name}'. "
            f"Available callable symbols include: {', '.join(available[:40])}"
        )
    model_fn = getattr(tv_models, model_name)
    if pretrained:
        kwargs.setdefault("pretrained", True)
    kwargs.setdefault("num_classes", int(spec.num_classes))
    return _build_with_pruning(model_fn, kwargs)


def _load_torchtext_model(spec: ModelSpec):
    try:
        import torchtext
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "torchtext backend requested but torchtext is not installed. "
            "Install with: pip install torchtext"
        ) from e
    if not hasattr(torchtext, "models"):
        version = getattr(torchtext, "__version__", "unknown")
        raise RuntimeError(
            "torchtext backend requires `torchtext.models`, which is unavailable in "
            f"installed torchtext=={version}. "
            "Use a newer torchtext build exposing `torchtext.models`, or use "
            "`model.backend=custom`/`model.backend=hf`."
        )
    tt_models = torchtext.models

    kwargs = _filtered_model_kwargs(spec)
    model_name = str(spec.name).strip()
    pretrained = bool(spec.model_kwargs.get("pretrained", False))

    if not hasattr(tt_models, model_name):
        available = sorted(name for name in dir(tt_models) if not name.startswith("_"))
        raise ValueError(
            f"Unknown torchtext model/bundle '{model_name}'. "
            f"Available symbols include: {', '.join(available[:40])}"
        )
    target = getattr(tt_models, model_name)

    if hasattr(target, "get_model"):
        if pretrained:
            # Most bundles expose pretrained weights through get_model().
            try:
                return target.get_model(**kwargs)
            except TypeError:
                return target.get_model()
        if hasattr(target, "transform"):
            try:
                return target.get_model(**kwargs)
            except TypeError:
                return target.get_model()
    if callable(target):
        return target(**kwargs)

    raise ValueError(
        f"torchtext symbol '{model_name}' is not directly instantiable. "
        "Use a bundle/class exposing `get_model()` or a callable model class."
    )


def _load_torchaudio_model(spec: ModelSpec):
    try:
        import torchaudio.models as ta_models
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "torchaudio backend requested but torchaudio is not installed. "
            "Install with: pip install torchaudio"
        ) from e

    kwargs = _filtered_model_kwargs(spec)
    model_name = str(spec.name).strip()
    if not hasattr(ta_models, model_name):
        available = sorted(
            name
            for name in dir(ta_models)
            if callable(getattr(ta_models, name, None)) and not name.startswith("_")
        )
        raise ValueError(
            f"Unknown torchaudio model '{model_name}'. "
            f"Available callable symbols include: {', '.join(available[:40])}"
        )
    model_fn = getattr(ta_models, model_name)
    if not callable(model_fn):
        raise ValueError(
            f"torchaudio symbol '{model_name}' is not callable."
        )
    return model_fn(**kwargs)


def _is_hf_candidate(name: str) -> bool:
    return "/" in str(name)


def _is_torchvision_candidate(name: str) -> bool:
    try:
        import torchvision.models as tv_models
    except Exception:
        return False
    if hasattr(tv_models, "get_model"):
        try:
            return str(name) in set(tv_models.list_models())
        except Exception:
            return False
    return hasattr(tv_models, str(name))


def _is_torchtext_candidate(name: str) -> bool:
    try:
        import torchtext
    except Exception:
        return False
    if not hasattr(torchtext, "models"):
        return False
    tt_models = torchtext.models
    return hasattr(tt_models, str(name))


def _is_torchaudio_candidate(name: str) -> bool:
    try:
        import torchaudio.models as ta_models
    except Exception:
        return False
    return hasattr(ta_models, str(name))


def _resolve_source(spec: ModelSpec) -> str:
    source = spec.source.lower()
    if source == "custom":
        return "appfl"
    if source in {"hf", "torchvision", "torchtext", "torchaudio"}:
        return source
    if source != "auto":
        raise ValueError(
            "model.backend must be one of: auto, custom, hf, "
            "torchvision, torchtext, torchaudio"
        )

    if spec.name in MODEL_REGISTRY:
        return "appfl"
    if _is_torchvision_candidate(spec.name):
        return "torchvision"
    if _is_torchtext_candidate(spec.name):
        return "torchtext"
    if _is_torchaudio_candidate(spec.name):
        return "torchaudio"
    if _is_hf_candidate(spec.name):
        return "hf"

    raise ValueError(
        f"Unable to resolve model source for exact name '{spec.name}'. "
        "Set model.backend explicitly and provide exact backend model name/card."
    )


def load_model(
    cfg: Dict[str, Any],
    input_shape: Tuple[int, ...],
    num_classes: int,
):
    """Unified model factory with explicit backend controls.

    Backends:
    - ``custom``: local models in ``appfl.sim.models``.
    - ``torchvision``: exact torchvision model name via ``model.name``.
    - ``torchtext``: exact torchtext model/bundle symbol via ``model.name``.
    - ``torchaudio``: exact torchaudio model name via ``model.name``.
    - ``hf``: exact model card id via ``model.name``.

    Aliases are intentionally disabled to avoid implicit behavior.
    """
    spec = _parse_model_spec(cfg=cfg, input_shape=input_shape, num_classes=num_classes)
    source = _resolve_source(spec)

    if source == "appfl":
        return _load_appfl_model(spec)
    if source == "torchvision":
        return _load_torchvision_model(spec)
    if source == "torchtext":
        return _load_torchtext_model(spec)
    if source == "torchaudio":
        return _load_torchaudio_model(spec)
    if source == "hf":
        return _load_hf_model(spec)

    raise RuntimeError(f"Unsupported resolved model source: {source}")
