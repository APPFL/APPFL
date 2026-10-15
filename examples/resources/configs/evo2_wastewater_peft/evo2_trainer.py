"""
Client-side APPFL trainer for Evo2 PEFT federated fine-tuning (LoRA/QLoRA/IA3)
on wastewater metagenomics data. Loaded via `trainer_path` (server_fedavg.yaml).
Pairs with `evo2_aggregator.py` on the server side.

Gates training on an AIDRIN/DRISEE data-readiness check (`aidrin_checker.py`):
files exceeding `drisee_threshold` are excluded; if all files in a site are
excluded, it sends an `AIDRIN_SKIP_KEY` sentinel instead of adapter weights.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, OrderedDict, Union

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from appfl.algorithm.trainer.base_trainer import BaseTrainer


_evo2_demo_dir = os.environ.get("EVO2_PROJECT_DIR")
if _evo2_demo_dir:
    sys.path.insert(0, _evo2_demo_dir)
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from aidrin_checker import AIDRIN_SKIP_KEY, check_site_data  # noqa: E402
from peft_finetune import (  # noqa: E402
    BATCH_SIZE,
    DEFAULT_MAX_ROUNDS,
    GRAD_ACCUM,
    WEIGHT_DECAY,
    FastaChunkDataset,
    collate_fn,
    inject_lora,
    load_evo2_model,
    train_one_round,
)


class Evo2PEFTTrainer(BaseTrainer):
    def __init__(
        self,
        model=None,  # unused — Evo2 loaded internally
        loss_fn=None,
        metric=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        train_configs: DictConfig = DictConfig({}),
        logger: Optional[Any] = None,
        **kwargs,
    ):
        self.train_configs = train_configs
        self.logger = logger
        self.round = 0
        self._last_metrics: Optional[Dict[str, Any]] = None

        self.site = train_configs.site
        self.method = train_configs.method
        self.base_data = Path(train_configs.base_data)
        self.model_path = Path(train_configs.model_path)
        self.evo2_dir = Path(train_configs.evo2_dir)
        self.num_steps = int(getattr(train_configs, "num_local_steps", 100))
        self.chunk_size = int(getattr(train_configs, "chunk_size", 8192))
        self.max_chunks = int(getattr(train_configs, "max_chunks", 0)) or None
        self.max_files = int(getattr(train_configs, "max_files", 0)) or None
        self.exclude_files = list(getattr(train_configs, "exclude_files", []) or [])
        self.lr = float(getattr(train_configs, "lr", 2e-4))
        self._aidrin_csv = (
            Path(getattr(train_configs, "aidrin_csv", ""))
            if getattr(train_configs, "aidrin_csv", "")
            else None
        )
        self._drisee_threshold = float(
            getattr(train_configs, "drisee_threshold", 20.0)
        )
        self._debug_payload = bool(getattr(train_configs, "debug_payload", False))

        if self.logger:
            self.logger.info(f"[{self.site}] Loading Evo2 7B ...")
        self.evo2_model, self.tokenizer = load_evo2_model(
            self.model_path, self.evo2_dir
        )
        if self.logger:
            self.logger.info(
                f"[{self.site}] Injecting {self.method.upper()} adapters ..."
            )
        self.evo2_model, _ = inject_lora(self.evo2_model, method=self.method)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- AIDRIN data-readiness gate (runs before dataset is built) ---
        self._site_ready = True
        self._aidrin_report: Dict[str, Any] = {}
        fasta_dir = self.base_data / self.site / "fasta"
        if self._aidrin_csv is not None:
            new_excludes, _contaminated, _report = check_site_data(
                site=self.site,
                aidrin_csv=self._aidrin_csv,
                fasta_dir=fasta_dir,
                current_exclude=self.exclude_files,
                max_files=self.max_files,
                drisee_threshold=self._drisee_threshold,
                report_dir=self.base_data / "aidrin_results",
                logger=self.logger,
            )
            if new_excludes:
                self.exclude_files = list(set(self.exclude_files) | set(new_excludes))
            self._site_ready = _report.get("site_ready", True)
            self._aidrin_report = _report

        # Build training dataset and dataloader once (uses updated exclude_files)
        self._dataset = FastaChunkDataset(
            fasta_dir,
            chunk_size=self.chunk_size,
            max_chunks=self.max_chunks,
            max_files=self.max_files,
            exclude_files=self.exclude_files or None,
        )
        self._dataloader = DataLoader(
            self._dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=(self.device.type == "cuda"),
            collate_fn=lambda b: collate_fn(b, self.tokenizer),
        )

        # Optimizer + scheduler (live across all FL rounds)
        trainable = [p for p in self.evo2_model.model.parameters() if p.requires_grad]
        self._optimizer = torch.optim.AdamW(
            trainable, lr=self.lr, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95)
        )
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self._optimizer,
            T_max=DEFAULT_MAX_ROUNDS * self.num_steps,
            eta_min=self.lr * 0.1,
        )

        if self.logger:
            if not self._site_ready:
                self.logger.info(
                    f"[{self.site}/{self.method}] AIDRIN: all training data contaminated"
                    f" — site will skip training and notify server"
                )
            else:
                self.logger.info(
                    f"[{self.site}/{self.method}] ready — "
                    f"{len(self._dataset):,} training chunks"
                )

    def train(self, **kwargs):
        self.round += 1
        if not self._site_ready:
            if self.logger:
                self.logger.info(
                    f"[{self.site}/{self.method}] FL round {self.round} — "
                    f"SKIPPED (AIDRIN: all training data contaminated)"
                )
            self._last_metrics = {
                "round": self.round,
                "current_local_steps": 0,
                "avg_loss": None,
                "tokens_seen": 0,
                "compute_second_per_step": 0.0,
                "aidrin_skipped": True,
            }
            return
        if self.logger:
            self.logger.info(
                f"[{self.site}/{self.method}] FL round {self.round} | "
                f"{self.num_steps} steps"
            )
        metrics = train_one_round(
            model=self.evo2_model,
            dataloader=self._dataloader,
            optimizer=self._optimizer,
            scheduler=self._scheduler,
            device=self.device,
            round_num=self.round,
            max_steps=self.num_steps,
            grad_accum=GRAD_ACCUM,
        )
        self._last_metrics = {
            "round": metrics["round"],
            "current_local_steps": metrics["steps"],
            "avg_loss": metrics["avg_loss"],
            "tokens_seen": metrics["tokens_seen"],
            "compute_second_per_step": round(
                metrics["elapsed_s"] / max(metrics["steps"], 1), 4
            ),
            "aidrin_skipped": False,
        }
        if self.logger:
            self.logger.info(
                f"[{self.site}/{self.method}] round {self.round} done | "
                f"loss={metrics['avg_loss']:.4f}"
            )

    def get_parameters(self) -> "tuple[Dict[str, torch.Tensor], Dict[str, Any]]":
        """Return (adapter weights, round metadata), or a contamination sentinel
        instead of weights if site data failed AIDRIN."""
        metadata = dict(self._last_metrics or {})
        metadata["site"] = self.site
        metadata["method"] = self.method
        if not self._site_ready:
            return {AIDRIN_SKIP_KEY: torch.tensor(1.0)}, metadata
        if self._debug_payload:
            return {"debug": torch.tensor([1.0])}, metadata
        params = {
            name: param.detach().cpu().clone()
            for name, param in self.evo2_model.model.named_parameters()
            if param.requires_grad
        }
        return params, metadata

    def load_parameters(self, params: Union[Dict, OrderedDict, None]):
        """Apply FedAvg-merged adapter weights received from the server.
        An empty or None params means fresh start — clients keep their own init."""
        if not params:
            if self.logger:
                self.logger.info(
                    f"[{self.site}/{self.method}] empty init model from server "
                    f"— starting from fresh adapter init"
                )
            return
        with torch.no_grad():
            for name, param in self.evo2_model.model.named_parameters():
                if name in params:
                    param.copy_(params[name].to(param.device))
        if self.logger:
            self.logger.info(
                f"[{self.site}/{self.method}] loaded merged adapter "
                f"({len(params)} tensors) from server"
            )

    def get_sample_size(self) -> int:
        return len(self._dataset)
