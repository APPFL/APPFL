"""
Client-side APPFL trainer for Evo2 PEFT federated fine-tuning (LoRA / QLoRA /
IA3), plus the gRPC transient-error retry wrapper and AIDRIN-report upload
helper needed to run it against an unreliable public endpoint (e.g. an
APPFLx relay).

This module bundles three things that the first working version of this
client (`evo2_peft_trainer.py` + `run_client.py`) had split across two files:

1. `Evo2PEFTTrainer` — loads Evo2 7B, injects LoRA/QLoRA/IA3 adapters, and
   gates training on an AIDRIN/DRISEE data-readiness check
   (`aidrin_checker.check_site_data`) before the first round ever starts. The
   Evo2 7B base model never leaves the compute node — only the adapter
   weights (~20 MB) are exchanged with the server.
2. `grpc_retry()` — retries a gRPC call on transient errors (UNAVAILABLE,
   DEADLINE_EXCEEDED, or an INTERNAL error mentioning RST_STREAM). Pulled out
   here so any launcher script imports one retry implementation instead of
   redefining it.
3. `send_aidrin_report()` — flattens the trainer's AIDRIN report and sends it
   to the server via `invoke_custom_action`, wrapped in `grpc_retry`.

See `examples/grpc/run_client_evo2.py` for the ~20-line launcher this reduces
to, and `appfl.algorithm.aggregator.evo2_aggregator` for the server-side
counterpart that filters out clients whose `get_parameters()` returns the
AIDRIN-contamination sentinel.

FL round lifecycle:
  train()            -> one round of local fine-tuning via train_one_round()
  get_parameters()   -> flat adapter state dict sent to server
  load_parameters()  -> apply merged adapter received from server

Expected train_configs keys (set in client YAML):
  site            str    e.g. "site_anl_1"
  method          str    "lora" | "qlora" | "ia3"
  base_data       str    path to data root
  model_path      str    path to evo2_7b.pt
  evo2_dir        str    path to writable evo2 repo copy
  num_local_steps int    gradient steps per FL round  (default 100)
  chunk_size      int    sequence chunk size           (default 8192)
  lr              float  learning rate                 (default 2e-4)
  aidrin_csv      str    path to site's AIDRIN input CSV (optional — skips
                         the readiness gate entirely if unset)
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional, OrderedDict, Union

import grpc
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from appfl.algorithm.trainer.base_trainer import BaseTrainer

from .aidrin_checker import AIDRIN_SKIP_KEY, check_site_data
from .peft_finetune import (
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

# --------------------------------------------------------------------------
# gRPC retry + AIDRIN report upload — shared by any launcher using this trainer
# --------------------------------------------------------------------------

DEFAULT_MAX_RETRIES = 20
DEFAULT_RETRY_DELAY = 30  # seconds between retries


def grpc_retry(
    fn,
    label: str,
    logger: Optional[Any] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: int = DEFAULT_RETRY_DELAY,
):
    """
    Call `fn()`, retrying on transient gRPC errors: UNAVAILABLE,
    DEADLINE_EXCEEDED, or an INTERNAL error whose details mention RST_STREAM
    (a mid-stream connection drop, observed in practice against public gRPC
    relays like APPFLx). Any other `grpc.RpcError` is re-raised immediately,
    as is the final attempt's error once `max_retries` is exhausted.

    Note: this does not help against a non-transient failure such as an
    expired TLS certificate on the server — that surfaces as a handshake
    failure beneath the RPC call itself and will exhaust every retry.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except grpc.RpcError as e:
            code = e.code()
            details = e.details() or ""
            is_transient = code in (
                grpc.StatusCode.UNAVAILABLE,
                grpc.StatusCode.DEADLINE_EXCEEDED,
            ) or (code == grpc.StatusCode.INTERNAL and "RST_STREAM" in details)
            if not is_transient:
                raise
            msg = (
                f"[retry {attempt}/{max_retries}] {label} failed "
                f"({code.name}: {details}) — retrying in {retry_delay}s"
            )
            if logger:
                logger.warning(msg)
            else:
                print(msg)
            if attempt == max_retries:
                raise
            time.sleep(retry_delay)


def send_aidrin_report(
    client_comm,
    aidrin_report: Dict[str, Any],
    logger: Optional[Any] = None,
    **retry_kwargs,
) -> None:
    """
    Flatten an `Evo2PEFTTrainer`'s AIDRIN report and send it to the server via
    `invoke_custom_action(action="get_data_readiness_report")`, retrying
    transient gRPC errors with `grpc_retry`. A no-op if `aidrin_report` is
    empty (e.g. the trainer was built with no `aidrin_csv` configured).

    Any `retry_kwargs` (e.g. `max_retries`, `retry_delay`) are forwarded to
    `grpc_retry`. A failure here is logged and swallowed rather than raised —
    losing the readiness report shouldn't abort an otherwise-working client.
    """
    if not aidrin_report:
        return
    flat = {
        "site": aidrin_report.get("site", ""),
        "n_clean": aidrin_report.get("n_clean", 0),
        "n_contaminated": aidrin_report.get("n_contaminated", 0),
        "site_ready": aidrin_report.get("site_ready", True),
        "drisee_threshold": aidrin_report.get("drisee_threshold", 20.0),
    }
    for k, v in aidrin_report.get("csv_meta_metrics", {}).items():
        if not k.startswith("_"):
            flat[f"csv_{k}"] = v
    try:
        grpc_retry(
            lambda: client_comm.invoke_custom_action(
                action="get_data_readiness_report", **flat
            ),
            "get_data_readiness_report",
            logger,
            **retry_kwargs,
        )
        if logger:
            logger.info("[AIDRIN] Data readiness report sent to server.")
    except Exception as e:
        if logger:
            logger.warning(f"[AIDRIN] Could not send readiness report: {e}")


# --------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------


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
        if self.logger:
            self.logger.info(
                f"[{self.site}/{self.method}] round {self.round} done | "
                f"loss={metrics['avg_loss']:.4f}"
            )

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Return adapter weights, or a contamination sentinel if site data failed AIDRIN."""
        if not self._site_ready:
            return {AIDRIN_SKIP_KEY: torch.tensor(1.0)}
        if self._debug_payload:
            return {"debug": torch.tensor([1.0])}
        return {
            name: param.detach().cpu().clone()
            for name, param in self.evo2_model.model.named_parameters()
            if param.requires_grad
        }

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
