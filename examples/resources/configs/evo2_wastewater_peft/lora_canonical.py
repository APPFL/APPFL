"""
Framework-independent names for Evo2-7B LoRA adapter tensors.

Clients built on different Evo2 implementations (the `evo2` pip package via
`evo2_trainer.py`, and Savanna via `savanna_trainer.py`) name the same layers
differently. Both trainers exchange adapters under these canonical keys so the
server's FedAvg can average them key-for-key:

    attn{slot}.{component}.{tensor}

slot      position of the layer in ATTN_LAYER_IDXS (0..4)
component wqkv | out_proj | mlp1 | mlp2 | mlp3
tensor    lora_A | lora_B | ia3_scale

lora_A is [rank, in_features] and lora_B is [out_features, rank] on both sides.
"""

import re

from peft_finetune import ATTN_LAYER_IDXS

EVO2_COMPONENTS = {
    "inner_mha_cls.Wqkv": "wqkv",
    "inner_mha_cls.out_proj": "out_proj",
    "mlp.l1": "mlp1",
    "mlp.l2": "mlp2",
    "mlp.l3": "mlp3",
}

SAVANNA_COMPONENTS = {
    "dense_projection": "wqkv",
    "dense": "out_proj",
    "w1": "mlp1",
    "w2": "mlp2",
    "w3": "mlp3",
}

_EVO2_TENSORS = {
    "lora_A.weight": "lora_A",
    "lora_B.weight": "lora_B",
    "scale": "ia3_scale",
}

_EVO2_PARAM_RE = re.compile(
    r"^blocks\.(\d+)\.(.+)\.(lora_A\.weight|lora_B\.weight|scale)$"
)


def canonical_key(layer_idx: int, component: str, tensor: str) -> str:
    if layer_idx not in ATTN_LAYER_IDXS:
        raise KeyError(
            f"layer {layer_idx} carries an adapter but is not one of the "
            f"attention layers {ATTN_LAYER_IDXS}"
        )
    return f"attn{ATTN_LAYER_IDXS.index(layer_idx)}.{component}.{tensor}"


def evo2_to_canonical(param_name: str) -> str:
    """`blocks.3.inner_mha_cls.Wqkv.lora_A.weight` -> `attn0.wqkv.lora_A`"""
    m = _EVO2_PARAM_RE.match(param_name)
    if not m or m.group(2) not in EVO2_COMPONENTS:
        raise KeyError(f"not an Evo2 adapter parameter: {param_name}")
    return canonical_key(
        int(m.group(1)), EVO2_COMPONENTS[m.group(2)], _EVO2_TENSORS[m.group(3)]
    )
