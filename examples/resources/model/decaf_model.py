"""
CLIP + LoRA model for dLoRA AB_SVD federated experiments.

Loads a CLIP backbone and injects LoRA layers into the specified
attention projections, returning the modified CLIP model ready for
federated fine-tuning with DeCaFTrainer.

Model kwargs (all optional, defaults match clip_lora/run_utils.py):
    backbone      (str):        CLIP backbone name.  Default: "ViT-B/16".
    encoder       (str):        "vision" | "text" | "both". Default: "both".
    position      (str):        Which transformer layers to inject LoRA.
                                Default: "all".
    params        (list[str]):  Attention matrices to patch.
                                Default: ["q", "k", "v"].
    r             (int):        LoRA rank.           Default: 2.
    alpha         (int):        LoRA scaling alpha.  Default: 1.
    dropout_rate  (float):      Dropout before LoRA. Default: 0.25.
    clip_lora_root (str):       Absolute path to the clip_lora project
                                directory.  Required so that the loralib
                                package (clip_lora/loralib/) can be found.
"""

import sys
import os


def DeCaFModel(
    backbone: str = "ViT-B/16",
    encoder: str = "both",
    position: str = "all",
    params=None,
    r: int = 2,
    alpha: int = 1,
    dropout_rate: float = 0.25,
    clip_lora_root: str = None,
    **kwargs,
):
    """
    Build and return a CLIP model with LoRA layers injected.

    The returned model contains all original CLIP parameters (frozen
    by default) plus the trainable LoRA matrices.  The trainer
    (DeCaFTrainer) will call mark_only_lora_as_trainable on the
    first training round.

    Args:
        backbone:       CLIP model name (e.g. "ViT-B/16", "ViT-B/32").
        encoder:        Which encoder(s) to inject LoRA into.
        position:       Which transformer layers to patch.
        params:         List of attention projection names to patch.
        r:              LoRA rank.
        alpha:          LoRA alpha scaling factor.
        dropout_rate:   Dropout rate applied before LoRA projection.
        clip_lora_root: Path to clip_lora project root.  This must be
                        set so that clip_lora/loralib can be imported.

    Returns:
        torch.nn.Module: CLIP model with LoRA layers applied.
    """
    if params is None:
        params = ["q", "k", "v"]

    # Bundled clip/loralib/datasets packages live at resources/clip_lora/
    # relative to this file (examples/resources/model/ -> ../clip_lora/).
    # An explicit clip_lora_root arg is accepted for backwards-compatibility
    # but is no longer required.
    _bundled = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "clip_lora")
    )
    for _p in filter(None, [_bundled, clip_lora_root]):
        if os.path.isdir(_p) and _p not in sys.path:
            sys.path.insert(0, _p)

    import clip
    from loralib.utils import apply_lora

    # Load base CLIP model (CPU at construction time; trainer moves to device)
    clip_model, _ = clip.load(backbone, device="cpu")
    clip_model.eval()

    # Build a minimal args-like namespace for apply_lora
    class _Args:
        pass

    args = _Args()
    args.encoder = encoder
    args.position = position
    args.backbone = backbone
    args.params = params
    args.r = r
    args.alpha = alpha
    args.dropout_rate = dropout_rate

    apply_lora(args, clip_model)

    return clip_model
