"""
DeCaFTrainer: Client-side trainer for federated CLIP + LoRA fine-tuning.

Extends VanillaTrainer to handle CLIP's dual-encoder architecture and
LoRA-specific parameter management.  The model passed in must be a CLIP
model with LoRA layers already applied (via the model file at
examples/resources/model/decaf_model.py).

Key differences from VanillaTrainer:
  - Freezes all non-LoRA parameters once (on first train() call).
  - Builds text features from classnames + template for the cosine-
    similarity loss; caches them when the text encoder is frozen.
  - Uses AdamW + CosineAnnealingLR over num_local_steps per round.
  - Uses torch.amp mixed-precision (fp16) matching clip_lora's original.
  - get_parameters() returns the full model state dict so the server
    aggregator can apply AB_SVD to the LoRA matrices while leaving
    frozen parameters unchanged via standard weighted average.

Required train_configs keys:
    classnames (list[str]): class labels for the dataset.
    template   (list[str]): e.g. ["a photo of a {}."]
    mode       (str): must be "step".
    num_local_steps (int): local training steps per FL round.

Optional train_configs keys:
    encoder        (str):   "vision" | "text" | "both". Default: "both".
    freeze_a       (bool):  freeze w_lora_A matrices.    Default: False.
    logit_scale    (float): CLIP cosine-sim scaling.     Default: 100.0.
    lr             (float): AdamW learning rate.         Default: 2e-4.
    weight_decay   (float): AdamW weight decay.          Default: 1e-2.
    device         (str):   training device.             Default: "cuda".
"""

import copy
import csv
import os
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from typing import Optional, Any, Dict, Union, OrderedDict
from torch.utils.data import Dataset, DataLoader

from appfl.algorithm.trainer.vanilla_trainer import VanillaTrainer


class DeCaFTrainer(VanillaTrainer):
    """
    Trainer for federated CLIP + LoRA fine-tuning (dLoRA AB_SVD).

    The model must be a CLIP model with LoRA layers already injected.
    All non-LoRA parameters are frozen on the first train() call.
    """

    def __init__(
        self,
        model=None,
        loss_fn=None,
        metric=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        train_configs: DictConfig = DictConfig({}),
        logger: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            metric=metric,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_configs=train_configs,
            logger=logger,
            **kwargs,
        )
        self._lora_marked = False  # whether LoRA-only freeze has been applied
        self._text_features_cache = None  # cached text features (vision-only training)
        self._scaler = torch.amp.GradScaler()

        # Re-create the train DataLoader with a seeded generator so that
        # batch order is deterministic when torch.manual_seed() is set by
        # the MPI launch script.  VanillaTrainer builds the loader without
        # a generator, which makes shuffle non-deterministic across runs.
        if self.train_dataset is not None:
            _g = torch.Generator()
            _g.manual_seed(torch.initial_seed())
            from torch.utils.data import DataLoader

            self.train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.train_configs.get("train_batch_size", 32),
                shuffle=self.train_configs.get("train_data_shuffle", True),
                num_workers=self.train_configs.get("num_workers", 0),
                generator=_g,
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def train(self, **kwargs):
        """Run one FL round of local CLIP + LoRA training."""
        if "round" in kwargs:
            self.round = kwargs["round"]

        # Freeze non-LoRA parameters once
        if not self._lora_marked:
            self._freeze_non_lora()
            self._lora_marked = True

        # Read training config
        encoder = self.train_configs.get("encoder", "both")
        # Use the model's learned logit_scale (clamped to CLIP's safe range)
        # rather than a fixed config value — 100.0 overflows fp16 gradients.
        if hasattr(self.model, "logit_scale"):
            logit_scale = self.model.logit_scale.exp().item()
        else:
            logit_scale = float(self.train_configs.get("logit_scale", 100.0))
        lr = float(self.train_configs.get("lr", 2e-4))
        weight_decay = float(self.train_configs.get("weight_decay", 1e-2))
        n_steps = int(
            self.train_configs.get("num_local_steps", len(self.train_dataloader))
        )
        device = self.train_configs.get("device", "cuda")

        # Classnames and template (from train_configs or dataset attributes)
        classnames = self.train_configs.get("classnames", None)
        template = self.train_configs.get("template", ["a photo of a {}."])
        if classnames is None and hasattr(self.train_dataset, "classnames"):
            classnames = self.train_dataset.classnames
        if isinstance(template, str):
            template = [template]
        if hasattr(self.train_dataset, "template"):
            template = self.train_dataset.template

        if classnames is None:
            raise ValueError(
                "DeCaFTrainer requires 'classnames' in train_configs "
                "or as an attribute of the train_dataset."
            )

        # Move model to device
        self.model.to(device)
        self.model.train()

        # Precompute text features (cache when text encoder is frozen)
        text_features = self._get_text_features(classnames, template, encoder, device)

        # Optimizer and scheduler (re-created each round, matching VanillaTrainer)
        lora_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            lora_params,
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_steps, eta_min=1e-6
        )

        # Training loop
        data_iter = iter(self.train_dataloader)
        total_loss = 0.0
        for _ in range(n_steps):
            try:
                images, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_dataloader)
                images, targets = next(data_iter)

            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            # Encode text features (re-encode when text encoder is trainable)
            if encoder in ("text", "both"):
                text_features = self._encode_text(classnames, template, device)

            # Encode image features
            if encoder in ("vision", "both"):
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    image_features = self.model.encode_image(images)
            else:
                with torch.no_grad():
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        image_features = self.model.encode_image(images)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            cosine_sim = logit_scale * image_features @ text_features.t()
            loss = F.cross_entropy(cosine_sim.float(), targets)

            self._scaler.scale(loss).backward()
            self._scaler.step(optimizer)
            scale_before = self._scaler.get_scale()
            self._scaler.update()
            # Only advance LR schedule when the optimizer actually stepped
            # (scaler skips the step when fp16 gradients overflow, in which
            # case the scale is reduced; stepping the scheduler then would
            # trigger a PyTorch warning and waste an LR decay tick).
            if self._scaler.get_scale() >= scale_before:
                scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / n_steps
        if self.logger:
            self.logger.info(
                f"[DeCaFTrainer] Round {self.round} — "
                f"avg loss: {avg_loss:.4f} over {n_steps} steps"
            )

        # Optional per-round validation
        acc = None
        if (
            self.train_configs.get("do_validation", False)
            and self.val_dataset is not None
        ):
            acc = self._validate(classnames, template, device, logit_scale)
            if self.logger:
                self.logger.info(
                    f"[DeCaFTrainer] Round {self.round} — val accuracy: {acc:.2f}%"
                )

        self.round += 1
        self.model_state = copy.deepcopy(self.model.state_dict())

        # ── Persist last checkpoint and accuracy log ──────────────────────
        output_dir = self.train_configs.get("logging_output_dirname", "./output")
        client_name = getattr(self, "client_id", None) or "client"
        os.makedirs(output_dir, exist_ok=True)

        # Overwrite single checkpoint file with current LoRA params
        ckpt_path = os.path.join(output_dir, f"checkpoint_{client_name}.pt")
        torch.save(
            {
                "round": self.round,
                "lora_state_dict": {
                    k: v.detach().cpu()
                    for k, v in self.model.state_dict().items()
                    if "lora_" in k
                },
            },
            ckpt_path,
        )

        # Append one row per round to a per-client CSV
        if acc is not None:
            csv_path = os.path.join(output_dir, f"accuracy_{client_name}.csv")
            write_header = not os.path.exists(csv_path)
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["round", "val_accuracy_pct", "avg_loss"])
                writer.writerow([self.round, f"{acc:.4f}", f"{avg_loss:.4f}"])

    def get_parameters(self) -> Dict:
        """Return only the LoRA parameters (w_lora_A / w_lora_B tensors).

        Sending only the trainable LoRA parameters (~720KB) instead of the
        full model state dict (~600MB) reduces communication and aggregation
        memory by ~800x per client.
        """
        return {
            k: v.detach().cpu()
            for k, v in self.model.state_dict().items()
            if "lora_" in k
        }

    def load_parameters(
        self,
        params: Union[Dict, OrderedDict, Any],
    ):
        """Load aggregated global model parameters."""
        self.model.load_state_dict(params, strict=False)
        # Invalidate cached text features in case the text encoder changed
        self._text_features_cache = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _validate(self, classnames, template, device, logit_scale) -> float:
        """Compute top-1 accuracy on the validation set. Returns accuracy in %."""
        self.model.eval()
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=int(self.train_configs.get("val_batch_size", 64)),
            shuffle=False,
            num_workers=0,
        )
        # Build text features once
        text_features = self._encode_text(classnames, template, device)

        correct = 0
        total = 0
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                image_features = self.model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = logit_scale * image_features @ text_features.t()
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        self.model.train()
        return 100.0 * correct / total if total > 0 else 0.0

    def _freeze_non_lora(self):
        """Freeze all parameters that do not belong to LoRA layers."""
        freeze_a = self.train_configs.get("freeze_a", False)
        for name, param in self.model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False
            elif freeze_a and "w_lora_A" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if self.logger:
            self.logger.info(
                f"[DeCaFTrainer] Frozen non-LoRA params. "
                f"Trainable parameters: {trainable:,}"
            )

    def _get_text_features(self, classnames, template, encoder, device):
        """
        Return (possibly cached) text feature matrix of shape (num_classes, D).
        Features are cached only when the text encoder is frozen (vision-only).
        """
        if encoder == "vision" and self._text_features_cache is not None:
            return self._text_features_cache

        features = self._encode_text(classnames, template, device)

        if encoder == "vision":
            self._text_features_cache = features
        return features

    def _encode_text(self, classnames, template, device):
        """Encode class names into normalised text feature vectors."""
        import clip as clip_pkg

        tmpl = template[0] if template else "a photo of a {}."
        texts = [tmpl.format(c.replace("_", " ")) for c in classnames]

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.float16):
                tokens = clip_pkg.tokenize(texts).to(device)
                class_embeddings = self.model.encode_text(tokens)
        text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        return text_features
