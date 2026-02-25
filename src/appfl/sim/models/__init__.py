"""Model zoo for appfl[sim] with lazy imports."""

from __future__ import annotations

import importlib


# Keep only custom models that are not delegated to external model hubs.
MODEL_REGISTRY = {
    "CelebACNN": ("celebacnn", "CelebACNN"),
    "DANet": ("danet", "DANet"),
    "FEMNISTCNN": ("femnistcnn", "FEMNISTCNN"),
    "LeNet": ("lenet", "LeNet"),
    "LogReg": ("logreg", "LogReg"),
    "M5": ("m5", "M5"),
    "Sent140LSTM": ("sent140lstm", "Sent140LSTM"),
    "SimpleCNN": ("simplecnn", "SimpleCNN"),
    "StackedGRU": ("stackedgru", "StackedGRU"),
    "StackedLSTM": ("stackedlstm", "StackedLSTM"),
    "StackedTransformer": ("stackedtransformer", "StackedTransformer"),
    "TwoCNN": ("twocnn", "TwoCNN"),
    "TwoNN": ("twonn", "TwoNN"),
    "UNet3D": ("unet3d", "UNet3D"),
}


def get_model_class(model_name: str):
    if model_name not in MODEL_REGISTRY:
        lower_map = {k.lower(): k for k in MODEL_REGISTRY}
        mapped = lower_map.get(model_name.lower())
        if mapped is not None:
            model_name = mapped
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Unsupported appfl model '{model_name}'. Available models: {available}"
        )

    module_name, class_name = MODEL_REGISTRY[model_name]
    module = importlib.import_module(f"appfl.sim.models.{module_name}")
    return getattr(module, class_name)


__all__ = ["MODEL_REGISTRY", "get_model_class"]
