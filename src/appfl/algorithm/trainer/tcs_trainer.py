"""Algorithm 1: SGD with trajectory consistency-based local stopping."""

import math

import torch
from torch.utils.data import DataLoader
from appfl.algorithm.trainer.base_trainer import BaseTrainer


class TCSTrainer(BaseTrainer):
    """APPFL trainer. Step mode uses num_base_steps / num_local_steps.

    Epoch mode fixes the reference after one full dataloader pass and uses
    num_local_epochs as the maximum; subsequent stopping is checked per batch.
    Only plain SGD implements the paper's average-gradient identity.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.train_configs
        if cfg.get("optim", "SGD") != "SGD":
            raise ValueError("TCS requires plain SGD")
        optim_args = dict(cfg.get("optim_args", {}))
        if any(
            optim_args.get(k, 0)
            for k in ("momentum", "weight_decay", "nesterov", "maximize")
        ):
            raise ValueError("TCS requires zero momentum/weight decay and descent SGD")
        if any(
            cfg.get(k, False)
            for k in ("use_dp", "clip_grad", "send_gradient", "use_secure_agg")
        ):
            raise ValueError("TCS expects unmodified SGD and full model parameters")
        self.lr = float(optim_args.get("lr", 0.1))
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be finite and positive")
        self.kappa = float(cfg.get("kappa", 0.5))
        if not 0 <= self.kappa < 1:
            raise ValueError("kappa must be in [0, 1)")
        self.device = torch.device(cfg.get("device", "cpu"))
        self.model.to(self.device)
        if isinstance(self.loss_fn, torch.nn.Module):
            self.loss_fn.to(self.device)
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if not self.parameters:
            raise ValueError("TCS requires trainable parameters")
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=cfg.get("train_batch_size", 32),
            shuffle=cfg.get("train_data_shuffle", True),
            drop_last=cfg.get("train_drop_last", False),
            num_workers=cfg.get("num_workers", 0),
        )
        if not len(self.train_dataloader):
            raise ValueError("TCS requires at least one training batch")
        mode = cfg.get("mode", "step")
        if mode == "step":
            self.base = self._positive_int(
                cfg.get("num_base_steps", 10), "num_base_steps"
            )
            self.cap = self._positive_int(
                cfg.get("num_local_steps", 100), "num_local_steps"
            )
        elif mode == "epoch":
            self.base = len(self.train_dataloader)
            epochs = self._positive_int(
                cfg.get("num_local_epochs", 1), "num_local_epochs"
            )
            self.cap = epochs * self.base
        else:
            raise ValueError("mode must be step or epoch")
        if self.cap < self.base:
            raise ValueError("local cap must be at least the base length")
        self.model_state = None
        self.metadata = None

    @staticmethod
    def _positive_int(value, name):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
        return value

    def train(self, **kwargs):
        round_id = kwargs.get("round", self.round)
        if isinstance(round_id, bool) or not isinstance(round_id, int) or round_id < 0:
            raise ValueError("round must be a nonnegative global round index")
        lr = float(kwargs.get("learning_rate", self.lr))
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("learning_rate must be finite and positive")
        cap = min(
            self.cap,
            self._positive_int(kwargs.get("local_steps", self.cap), "local_steps"),
        )
        if cap < self.base:
            raise ValueError("remaining work cannot accommodate the base length")
        if round_id == 0:
            cap = self.base
        self.model.train()
        optimizer = torch.optim.SGD(self.parameters, lr=lr)
        initial = [p.detach().clone() for p in self.parameters]
        iterator = iter(self.train_dataloader)
        reference = None
        psi = ratio = cosine = None
        reason = "base" if round_id == 0 else "cap"
        examples = 0
        loss_sum = 0.0
        for h in range(1, cap + 1):
            try:
                data, target = next(iterator)
            except StopIteration:
                iterator = iter(self.train_dataloader)
                data, target = next(iterator)
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad(set_to_none=True)
            loss = self.loss_fn(self.model(data), target)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite local loss")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())
            examples += len(target)
            if h < self.base:
                continue
            with torch.no_grad():
                direction = torch.cat(
                    [(a - p).reshape(-1) for a, p in zip(initial, self.parameters)]
                ) / (lr * h)
                if not torch.isfinite(direction).all():
                    raise FloatingPointError("non-finite local update")
                if reference is None:
                    reference = direction.clone()
                    ref_norm = torch.linalg.vector_norm(reference)
                norm = torch.linalg.vector_norm(direction)
                psi = float(self._score(direction, reference))
                ratio = float(norm / ref_norm) if ref_norm > 0 else None
                cosine = (
                    float(torch.dot(direction, reference) / (norm * ref_norm))
                    if norm > 0 and ref_norm > 0
                    else None
                )
            if round_id > 0 and psi <= 0:
                reason = "trajectory"
                break
        self.model_state = {
            k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
        }
        self.metadata = dict(
            round=round_id,
            num_steps=h,
            num_samples=len(self.train_dataset),
            learning_rate=lr,
            stop_reason=reason,
            psi=psi,
            norm_ratio=ratio,
            cosine=cosine,
            local_epochs=h / len(self.train_dataloader),
            examples_seen=examples,
            train_loss=loss_sum / h,
        )
        self.round = round_id + 1
        if self.logger is not None:
            self.logger.info(
                f"TCS round={round_id} steps={h} epochs={self.metadata['local_epochs']:.3f} stop={reason}"
            )

    def get_parameters(self):
        if self.model_state is None:
            return {
                k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
            }
        return self.model_state, dict(self.metadata)

    def _score(self, direction, reference):
        return (
            torch.linalg.vector_norm(direction)
            - torch.linalg.vector_norm(direction - reference)
            - self.kappa * torch.linalg.vector_norm(reference)
        )

    def load_parameters(self, params):
        # APPFL's serial example passes the complete scheduler response, while
        # gRPC separates model and metadata before calling this method.
        round_id = self.round
        if isinstance(params, tuple):
            params, metadata = params
            round_id = metadata["round"]
            if (
                isinstance(round_id, bool)
                or not isinstance(round_id, int)
                or round_id < 0
            ):
                raise ValueError("invalid global round in model response")
        self.model.load_state_dict(params, strict=True)
        self.round = round_id
        self.model_state = self.metadata = None
