"""Compute model for the virtual-time async FL simulator (v2).

Replaces v1's simple ``cps * steps * compute_factor`` with multiple modes:

  measured  — v1 default: use VanillaTrainer's measured compute_second_per_step
  factor    — v1 with explicit lognormal factor (same formula, explicit sampling)
  profile   — per-device speed ratios from hardware benchmarks (A100=1x, RPi4=24000x)
  flops     — FLOPs-based estimation from model architecture + device TFLOPS

Backward compatible: mode="measured" reproduces v1 behavior exactly.
"""

from dataclasses import dataclass, field
from typing import Optional
import random as _random


DEVICE_PROFILES = {
    "a100":       {"tflops": 312,    "desc": "NVIDIA A100 (datacenter)"},
    "h100":       {"tflops": 756,    "desc": "NVIDIA H100 (datacenter)"},
    "v100":       {"tflops": 125,    "desc": "NVIDIA V100 (datacenter)"},
    "a10":        {"tflops": 62.5,   "desc": "NVIDIA A10 (inference)"},
    "rtx4090":    {"tflops": 82.6,   "desc": "NVIDIA RTX 4090 (consumer)"},
    "rtx3090":    {"tflops": 71,     "desc": "NVIDIA RTX 3090 (consumer)"},
    "rtx3080":    {"tflops": 29.8,   "desc": "NVIDIA RTX 3080 (consumer)"},
    "gtx1080":    {"tflops": 8.9,    "desc": "NVIDIA GTX 1080 (consumer)"},
    "gtx1060":    {"tflops": 4.4,    "desc": "NVIDIA GTX 1060 (consumer)"},
    "jetson_orin": {"tflops": 40,    "desc": "NVIDIA Jetson Orin (edge)"},
    "jetson_nx":  {"tflops": 21,     "desc": "NVIDIA Jetson Xavier NX (edge)"},
    "jetson_nano": {"tflops": 0.472, "desc": "NVIDIA Jetson Nano (edge)"},
    "m1":         {"tflops": 2.6,    "desc": "Apple M1 (laptop)"},
    "m2":         {"tflops": 3.6,    "desc": "Apple M2 (laptop)"},
    "rpi4":       {"tflops": 0.013,  "desc": "Raspberry Pi 4 CPU (IoT)"},
    "esp32":      {"tflops": 0.0004, "desc": "ESP32 microcontroller (IoT)"},
}

# Reference device: all speed ratios are relative to this
_REF_DEVICE = "a100"


@dataclass
class ComputeModel:
    """Per-client compute model."""
    mode: str = "measured"
    compute_factor: float = 1.0
    device_type: str = "a100"
    gpu_utilization: float = 0.5
    model_flops_per_step: float = 0.0

    def compute_time(self, cps: float, num_steps: int, **kwargs) -> float:
        """
        Compute virtual training time based on the configured mode.

        :param cps: Compute seconds per step (measured or fixed).
        :param num_steps: Number of local training steps.
        :return: Virtual compute duration in seconds.
        """
        if self.mode == "measured":
            return cps * num_steps * self.compute_factor
        elif self.mode == "factor":
            return cps * num_steps * self.compute_factor
        elif self.mode == "profile":
            ref = DEVICE_PROFILES[_REF_DEVICE]["tflops"]
            target = DEVICE_PROFILES.get(self.device_type, DEVICE_PROFILES[_REF_DEVICE])["tflops"]
            ratio = ref / max(target, 0.0001)
            return cps * num_steps * ratio
        elif self.mode == "tier":
            return cps * num_steps * self.compute_factor
        elif self.mode == "flops":
            if self.model_flops_per_step <= 0:
                return cps * num_steps * self.compute_factor
            batch_size = kwargs.get("batch_size", 64)
            total_flops = self.model_flops_per_step * num_steps * batch_size
            device_flops = DEVICE_PROFILES.get(self.device_type, DEVICE_PROFILES[_REF_DEVICE])["tflops"] * 1e12
            return total_flops / (device_flops * max(self.gpu_utilization, 0.01))
        return cps * num_steps * self.compute_factor


def build_compute_models(client_ids, compute_cfg, het_cfg, seed):
    """Build a ComputeModel per client from config.

    Returns {cid: ComputeModel}.  When compute_cfg is empty, falls back to
    het_cfg's compute.distribution for v1-style lognormal factor sampling.
    """
    rng = _random.Random(seed)

    mode = "measured"
    flops = 0.0
    utilization = 0.5

    if compute_cfg:
        mode = compute_cfg.get("mode", "measured")
        flops = compute_cfg.get("model_flops_per_step", 0.0)
        utilization = compute_cfg.get("gpu_utilization", 0.5)

    # device type sampling (for profile/flops modes)
    dev_cfg = compute_cfg.get("device_types", {}) if compute_cfg else {}
    dev_options = dev_cfg.get("options", [_REF_DEVICE])
    dev_weights = dev_cfg.get("weights", [1.0] * len(dev_options))

    # tier sampling (for tier mode)
    tiers_cfg = compute_cfg.get("tiers", []) if compute_cfg else []

    # compute_factor sampling (for measured/factor modes)
    comp_het = het_cfg.get("compute", {}) if het_cfg else {}

    models = {}
    for cid in client_ids:
        # sample compute_factor
        if mode == "tier" and tiers_cfg:
            tier_weights = [t.get("proportion", 1.0) for t in tiers_cfg]
            total = sum(tier_weights)
            r = rng.random() * total
            cumul = 0
            chosen = tiers_cfg[0]
            for t, w in zip(tiers_cfg, tier_weights):
                cumul += w
                if r <= cumul:
                    chosen = t
                    break
            cf = chosen.get("factor", 1.0)
            dev = chosen.get("name", "tier")
        elif comp_het.get("distribution") == "lognormal":
            pr = comp_het.get("params", {})
            cf = rng.lognormvariate(pr.get("mu", 0.0), pr.get("sigma", 0.5))
            dev = _REF_DEVICE
        else:
            cf = 1.0
            dev = _REF_DEVICE

        # sample device type (for profile/flops modes)
        if mode in ("profile", "flops") and dev_options:
            total = sum(dev_weights)
            r = rng.random() * total
            cumul = 0
            dev = dev_options[0]
            for opt, w in zip(dev_options, dev_weights):
                cumul += w
                if r <= cumul:
                    dev = opt
                    break

        models[cid] = ComputeModel(
            mode=mode,
            compute_factor=cf,
            device_type=dev,
            gpu_utilization=utilization,
            model_flops_per_step=flops,
        )

    return models
