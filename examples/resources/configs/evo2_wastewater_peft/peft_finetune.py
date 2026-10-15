#!/usr/bin/env python3
"""
peft_finetune.py

DB-FORGE -- Phase 4: Federated PEFT Fine-tuning

Runs local PEFT fine-tuning of Evo2 7B on a single site's wastewater
metagenomics FASTA/FASTQ data. Called independently per site -- raw
sequences never leave the site. Only adapter weights are exported for
aggregation.

Three additions over the original pilot run:
  1. Held-out split    -- within-site samples excluded from training,
                          evaluated locally after every round
  2. Convergence stop  -- training stops when training loss delta falls
                          below --min-delta for --patience rounds, instead
                          of running a fixed number of rounds
  3. Federated eval     -- after FedAvg the merged adapter is sent back
                          and evaluated on held-out data locally; only
                          scalar metrics (loss, PPL) cross site boundaries

Supported PEFT methods (one script invocation can run several in
sequence -- see --method below):
  lora   -- low-rank matrix pairs on attention + MLP linear layers
  qlora  -- 4-bit quantized base + LoRA adapters (75% less GPU memory)
  ia3    -- learned element-wise rescaling vectors (26x smaller adapters)

Target layers (attention blocks; indices are an Evo2-7B architecture
property, the same regardless of site or data -- not CLI-configurable):
  inner_mha_cls.Wqkv     4096 -> 12288  (combined QKV projection)
  inner_mha_cls.out_proj 4096 -> 4096   (output projection)
  mlp.l1                 4096 -> 11264  (MLP gate)
  mlp.l2                 4096 -> 11264  (MLP up)
  mlp.l3                11264 -> 4096   (MLP down)

IMPORTANT -- chunk size depends on your data's sequence length regime,
and this script does NOT auto-detect it:
  --chunk-size       must match the actual sequence length regime of the
                      site being trained: 8192 for long assembled contigs,
                      150 (or your reads' actual length) for raw short
                      reads. Using 8192 on short-read data will not error,
                      but every chunk will just be the whole short read --
                      a different (untested for your data) regime than
                      either validated case in this study.
  --eval-chunk-size   same consideration, but for held-out evaluation.
                      Independent of --chunk-size; must be set separately.
  See README for the wastewater case study's two validated chunk-size /
  threshold regimes (8192bp contigs and 150bp short reads).

Configuring paths (environment-specific settings):
    --base-data /path/to/data           or  export DB_FORGE_BASE_DATA=...
    --model-path /path/to/evo2_7b.pt    or  export DB_FORGE_MODEL_PATH=...
    --evo2-dir /path/to/evo2/repo       or  export DB_FORGE_EVO2_DIR=...

Assumed environment (documentation only -- this script does not check for
or set up the environment itself; see setup_polaris_environment.sh):
    - GPU(s) required in practice (CPU will technically run but is
      infeasible for a 7B model)
    - PyTorch, and the Evo2 package importable from --evo2-dir

Usage:
  # Single method, run to convergence with held-out evaluation
  python peft_finetune.py --site site_anl_1 --method qlora \\
      --base-data /path/to/data --model-path /path/to/evo2_7b.pt \\
      --evo2-dir /path/to/evo2

  # Multiple methods in one invocation, run sequentially
  python peft_finetune.py --site site_anl_1 --method lora qlora ... \\
      --base-data /path/to/data --model-path /path/to/evo2_7b.pt --evo2-dir /path/to/evo2

  # All three methods
  python peft_finetune.py --site site_anl_1 --method all ...

  # Cap rounds for testing
  python peft_finetune.py --site site_anl_1 --method lora --max-rounds 5 ...


  # Short-read site (raw Illumina reads, not assembled contigs)
  python peft_finetune.py --site site_novel_1 --method qlora \\
      --chunk-size 150 --eval-chunk-size 150 ...

  # After FedAvg: evaluate merged adapter on held-out data
  python peft_finetune.py --site site_anl_1 --method lora \\
      --eval-only --merged-adapter path/to/merged.pt ...
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── CONFIG: training hyperparameters and architecture constants ─────────────
# These are validated defaults / architecture properties, not
# environment-specific paths -- all remain CLI-overridable where it makes
# sense (chunk size, rank, lr, rounds, etc.) but keep their defaults here.

CHUNK_SIZE           = 8192
BATCH_SIZE           = 4   # 4 sequences in parallel (~12-15 GB/GPU, safe on 40GB A100)
GRAD_ACCUM           = 2   # effective batch = 8 sequences
LEARNING_RATE        = 2e-4
WEIGHT_DECAY         = 0.01
MAX_STEPS_PER_ROUND  = 100
WARMUP_STEPS         = 10

DEFAULT_MAX_ROUNDS   = 20
DEFAULT_MIN_DELTA    = 0.005  # validated against loss scales of 2-6.5 in this
                              # study; a very different dataset may need a
                              # different value -- see README
DEFAULT_PATIENCE     = 2

LORA_RANK    = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05

# Attention layer indices in StripedHyena2 Evo2-7B -- an architecture
# property, identical regardless of site or data. Not CLI-configurable.
ATTN_LAYER_IDXS = [3, 10, 17, 24, 31]

EVAL_CHUNKS      = 50    # chunks per eval round
EVAL_CHUNK_SIZE  = 8192  # default; override with --eval-chunk-size for
                         # short-read sites -- see module docstring

ALL_METHODS = ["lora", "qlora", "ia3"]
# ─────────────────────────────────────────────────────────────────────────────

def resolve_path(cli_value: Optional[str], env_var: str, label: str) -> Path:
    """Resolve a required path from --cli-flag, then $ENV_VAR, else error."""
    if cli_value:
        return Path(cli_value)
    env_value = os.environ.get(env_var)
    if env_value:
        return Path(env_value)
    print(
        f"ERROR: {label} not set.\n"
        f"  Provide it via the corresponding CLI flag, or:\n"
        f"  export {env_var}=/path/to/...",
        file=sys.stderr,
    )
    sys.exit(1)

# ── DATASET ───────────────────────────────────────────────────────────────────

class FastaChunkDataset(Dataset):
    """
    Streams FASTA/FASTQ files, chunks into fixed-length windows.
    Accepts an explicit exclude_files list for held-out splitting.
    Data sovereignty: only reads from local site path.
    """

    VALID_BASES = set("ACGTNacgtn")

    def __init__(self, fasta_dir: Path, chunk_size: int = CHUNK_SIZE,
                 max_chunks: Optional[int] = None,
                 exclude_files: Optional[List[str]] = None,
                 max_files: Optional[int] = None):
        self.chunk_size = chunk_size
        self.chunks     = []
        self._load_fastas(fasta_dir, max_chunks, exclude_files or [], max_files)

    def _load_fastas(self, fasta_dir: Path, max_chunks: Optional[int],
                      exclude_files: List[str], max_files: Optional[int] = None):
        fasta_files = sorted(
            list(fasta_dir.glob("*.fna")) + list(fasta_dir.glob("*.fastq"))
        )
        if not fasta_files:
            raise FileNotFoundError(f"No .fna or .fastq files in {fasta_dir}")

        excluded = [f for f in fasta_files
                    if any(ex in f.name for ex in exclude_files)]
        included = [f for f in fasta_files
                    if not any(ex in f.name for ex in exclude_files)]

        if max_files:
            # Smallest-N-by-size, matching aidrin_checker.check_site_data's
            # candidate selection so the AIDRIN gate and the actual training
            # set agree on which files are in play.
            included = sorted(included, key=lambda f: f.stat().st_size)[:max_files]

        if excluded:
            print(f"   Held-out (excluded from training):")
            for f in excluded:
                print(f"     {f.name}")
        print(f"   Training files ({len(included)}):")

        for fasta_path in included:
            print(f"     {fasta_path.name}")
            is_fastq = fasta_path.suffix == ".fastq"
            seq_buf  = []
            line_num = 0
            with open(fasta_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if is_fastq:
                        line_num += 1
                        if line_num % 4 == 2:
                            self._add_sequence(line.upper())
                    else:
                        if line.startswith(">"):
                            if seq_buf:
                                self._add_sequence("".join(seq_buf))
                                seq_buf = []
                        else:
                            seq_buf.append(line.upper())
                if seq_buf:
                    self._add_sequence("".join(seq_buf))

            if max_chunks and len(self.chunks) >= max_chunks:
                self.chunks = self.chunks[:max_chunks]
                break

        print(f"   Training chunks: {len(self.chunks):,}  "
              f"(chunk_size={self.chunk_size})")

    def _add_sequence(self, seq: str):
        seq = "".join(c for c in seq if c in self.VALID_BASES)
        if len(seq) < 32:
            return
        if len(seq) < self.chunk_size:
            self.chunks.append(seq)
            return
        for start in range(0, len(seq) - self.chunk_size, self.chunk_size):
            self.chunks.append(seq[start:start + self.chunk_size])

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]

def load_eval_chunks(fasta_paths: List[Path], chunk_size: int,
                      n_chunks: int = EVAL_CHUNKS) -> List[str]:
    """Load eval chunks from held-out files. Handles FASTA and FASTQ."""
    valid  = set("ACGTN")
    chunks = []
    for fasta_path in fasta_paths:
        if not fasta_path.exists():
            print(f"   [WARN] Held-out file not found: {fasta_path}")
            continue
        is_fastq = fasta_path.suffix == ".fastq"
        seq_buf  = []
        line_num = 0
        with open(fasta_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if is_fastq:
                    line_num += 1
                    if line_num % 4 == 2:
                        seq = "".join(c for c in line.upper() if c in valid)
                        if len(seq) < 32:
                            continue
                        if len(seq) < chunk_size:
                            chunks.append(seq)
                        else:
                            for start in range(0, len(seq) - chunk_size, chunk_size):
                                chunks.append(seq[start:start + chunk_size])
                        if len(chunks) >= n_chunks:
                            return chunks
                else:
                    if line.startswith(">"):
                        if seq_buf:
                            seq = "".join(seq_buf).upper()
                            seq = "".join(c for c in seq if c in valid)
                            if len(seq) < 32:
                                seq_buf = []
                                continue
                            if len(seq) < chunk_size:
                                chunks.append(seq)
                            else:
                                for start in range(0, len(seq) - chunk_size, chunk_size):
                                    chunks.append(seq[start:start + chunk_size])
                            if len(chunks) >= n_chunks:
                                return chunks
                        seq_buf = []
                    else:
                        seq_buf.append(line)
        if len(chunks) >= n_chunks:
            break
    return chunks[:n_chunks]

def collate_fn(batch: List[str], tokenizer) -> Dict[str, torch.Tensor]:
    tokenized = [torch.tensor(tokenizer.tokenize(seq), dtype=torch.long)
                 for seq in batch]
    max_len   = max(t.shape[0] for t in tokenized)
    pad_token = 0
    padded    = torch.stack([
        torch.nn.functional.pad(t, (0, max_len - t.shape[0]), value=pad_token)
        for t in tokenized
    ])
    labels = padded.clone()
    for i, t in enumerate(tokenized):
        if t.shape[0] < max_len:
            labels[i, t.shape[0]:] = -100
    return {"input_ids": padded, "labels": labels}

# ── LORA ──────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """LoRA adapter -- additive low-rank update: output += B @ A @ x * scaling"""

    def __init__(self, base_layer: nn.Module, rank: int = LORA_RANK,
                 alpha: float = LORA_ALPHA, dropout: float = LORA_DROPOUT):
        super().__init__()
        self.base_layer = base_layer
        self.rank       = rank
        self.scaling    = alpha / rank
        self.dropout    = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        in_f, out_f = self._get_dims(base_layer)
        self.lora_A = nn.Linear(in_f, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_f, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        for param in self.base_layer.parameters():
            param.requires_grad = False

    def _get_dims(self, layer) -> Tuple[int, int]:
        if hasattr(layer, "weight"):
            return layer.weight.shape[1], layer.weight.shape[0]
        elif hasattr(layer, "in_features") and hasattr(layer, "out_features"):
            return layer.in_features, layer.out_features
        raise ValueError(f"Cannot determine dims for {type(layer)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        x_cast   = x.to(self.lora_A.weight.dtype)
        lora_out = self.lora_B(self.lora_A(self.dropout(x_cast))).to(base_out.dtype)
        return base_out + lora_out * self.scaling

    def get_adapter_state_dict(self) -> Dict:
        return {
            "lora_A":  self.lora_A.weight.detach().cpu(),
            "lora_B":  self.lora_B.weight.detach().cpu(),
            "rank":    self.rank,
            "scaling": self.scaling,
        }

# ── IA3 ───────────────────────────────────────────────────────────────────────

class IA3Linear(nn.Module):
    """IA3 adapter -- multiplicative rescaling: output = base_output * scale"""

    def __init__(self, base_layer: nn.Module, is_feedforward: bool = False):
        super().__init__()
        self.base_layer     = base_layer
        self.is_feedforward = is_feedforward
        out_f               = self._get_out_features(base_layer)
        self.scale          = nn.Parameter(torch.ones(out_f))

        for param in self.base_layer.parameters():
            param.requires_grad = False

    def _get_out_features(self, layer) -> int:
        if hasattr(layer, "weight"):
            return layer.weight.shape[0]
        elif hasattr(layer, "out_features"):
            return layer.out_features
        raise ValueError(f"Cannot determine out_features for {type(layer)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base_layer(x)
        return out * self.scale.to(out.dtype)

    def get_adapter_state_dict(self) -> Dict:
        return {"scale": self.scale.detach().cpu()}

# ── ADAPTER INJECTION ─────────────────────────────────────────────────────────

def inject_lora(model: nn.Module, method: str) -> Tuple[nn.Module, int]:
    """Inject LoRA or IA3 into Evo2 attention blocks. Moves adapters to correct device."""
    base_model = model.model if hasattr(model, "model") else model
    n_injected = 0

    for layer_idx in ATTN_LAYER_IDXS:
        block_name = f"blocks.{layer_idx}"
        try:
            block = base_model.get_submodule(block_name)
        except Exception:
            print(f"   [WARN] {block_name} not found")
            continue

        mha = block.inner_mha_cls

        if hasattr(mha, "Wqkv"):
            try:
                dev = next(mha.Wqkv.parameters()).device
                mha.Wqkv = (LoRALinear(mha.Wqkv) if method in ("lora", "qlora")
                            else IA3Linear(mha.Wqkv)).to(dev)
                n_injected += 1
                print(f"   [{dev}] {method} -> {block_name}.inner_mha_cls.Wqkv")
            except Exception as e:
                print(f"   [WARN] {block_name}.Wqkv: {e}")

        if hasattr(mha, "out_proj"):
            try:
                dev = next(mha.out_proj.parameters()).device
                mha.out_proj = (LoRALinear(mha.out_proj) if method in ("lora", "qlora")
                                else IA3Linear(mha.out_proj)).to(dev)
                n_injected += 1
                print(f"   [{dev}] {method} -> {block_name}.inner_mha_cls.out_proj")
            except Exception as e:
                print(f"   [WARN] {block_name}.out_proj: {e}")

        mlp = block.mlp
        for lname in ["l1", "l2", "l3"]:
            layer = getattr(mlp, lname, None)
            if layer is None:
                continue
            try:
                dev = next(layer.parameters()).device
                if method in ("lora", "qlora"):
                    setattr(mlp, lname, LoRALinear(layer).to(dev))
                else:
                    setattr(mlp, lname,
                            IA3Linear(layer, is_feedforward=(lname == "l1")).to(dev))
                n_injected += 1
                print(f"   [{dev}] {method} -> {block_name}.mlp.{lname}")
            except Exception as e:
                print(f"   [WARN] {block_name}.mlp.{lname}: {e}")

    base_model2 = model.model if hasattr(model, "model") else model
    trainable = sum(p.numel() for p in base_model2.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in base_model2.parameters())
    print(f"\n   Adapters injected: {n_injected}")
    print(f"   Trainable params:  {trainable:,} / {total:,} "
          f"({100*trainable/total:.3f}%)")
    return model, trainable

def apply_merged_adapter(model: nn.Module, merged_path: Path, method: str):
    """Load a FedAvg-merged adapter into an already-injected model."""
    pkg     = torch.load(merged_path, map_location="cpu")
    merged  = pkg["adapters"]
    base    = model.model if hasattr(model, "model") else model
    applied = 0

    for layer_path, weights in merged.items():
        parts       = layer_path.split(".")
        parent_path = ".".join(parts[:-1])
        attr_name   = parts[-1]
        try:
            parent = base.get_submodule(parent_path)
            module = getattr(parent, attr_name)
            if isinstance(module, LoRALinear):
                dev = module.lora_A.weight.device
                module.lora_A.weight.data = weights["lora_A"].to(dev)
                module.lora_B.weight.data = weights["lora_B"].to(dev)
                applied += 1
            elif isinstance(module, IA3Linear):
                dev = module.scale.device
                module.scale.data = weights["scale"].to(dev)
                applied += 1
        except Exception as e:
            print(f"   [WARN] Could not apply {layer_path}: {e}")

    print(f"   Applied {applied} merged adapter layers from {merged_path.name}")
    return model

# ── LOSS / EVAL ───────────────────────────────────────────────────────────────

def compute_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )

@torch.no_grad()
def evaluate(model: nn.Module, eval_chunks: List[str],
             tokenizer, device: torch.device) -> Tuple[float, float]:
    """Evaluate on held-out chunks. Only scalar metrics leave the site."""
    base = model.model if hasattr(model, "model") else model
    base.eval()
    losses = []

    for seq in eval_chunks:
        ids = torch.tensor(tokenizer.tokenize(seq),
                            dtype=torch.long).unsqueeze(0).to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out    = base.forward(ids)
            logits = out[0] if isinstance(out, tuple) else out
        loss = compute_loss(logits, ids)
        losses.append(loss.item())

    mean_loss = sum(losses) / len(losses)
    ppl       = float(torch.exp(torch.tensor(mean_loss)))
    base.train()
    return round(mean_loss, 6), round(ppl, 2)

# ── TRAINING ──────────────────────────────────────────────────────────────────

def train_one_round(model, dataloader, optimizer, scheduler,
                     device, round_num, max_steps, grad_accum) -> Dict:
    """Train for one federated round."""
    base = model.model if hasattr(model, "model") else model
    base.train()

    for param in base.parameters():
        param.requires_grad = False
    for name, module in base.named_modules():
        if isinstance(module, LoRALinear):
            module.lora_A.weight.requires_grad = True
            module.lora_B.weight.requires_grad = True
        elif isinstance(module, IA3Linear):
            module.scale.requires_grad = True

    for param in base.parameters():
        if param.requires_grad:
            param.grad = None

    total_loss  = 0.0
    step        = 0
    tokens_seen = 0
    t_start     = time.time()
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        if step >= max_steps:
            break

        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out    = model.model.forward(input_ids)
            logits = out[0] if isinstance(out, tuple) else out
            loss   = compute_loss(logits, labels) / grad_accum

        loss.backward()
        total_loss  += loss.item() * grad_accum
        tokens_seen += input_ids.numel()

        if (batch_idx + 1) % grad_accum == 0:
            _base = model.model if hasattr(model, "model") else model
            torch.nn.utils.clip_grad_norm_(
                [p for p in _base.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

            if step % 10 == 0:
                print(f"   Round {round_num} | step {step:>4}/{max_steps} | "
                      f"loss={total_loss/step:.4f} | "
                      f"lr={scheduler.get_last_lr()[0]:.2e} | "
                      f"{tokens_seen/1e6:.1f}M tokens | "
                      f"{time.time()-t_start:.0f}s")

    return {
        "round":       round_num,
        "steps":       step,
        "avg_loss":    round(total_loss / max(step, 1), 6),
        "tokens_seen": tokens_seen,
        "elapsed_s":   round(time.time() - t_start, 1),
    }

# ── ADAPTER EXPORT ────────────────────────────────────────────────────────────

def export_adapters(model, output_path, site, method, round_num,
                     train_metrics, val_metrics=None) -> Path:
    """Export adapter weights only. Base model never saved."""
    adapters = {}
    base     = model.model if hasattr(model, "model") else model
    for name, module in base.named_modules():
        if isinstance(module, (LoRALinear, IA3Linear)):
            adapters[name] = module.get_adapter_state_dict()

    torch.save({
        "site":        site,
        "method":      method,
        "round":       round_num,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "lora_rank":   LORA_RANK if method in ("lora", "qlora") else None,
        "n_adapters":  len(adapters),
        "metrics":     train_metrics,
        "val_metrics": val_metrics,
        "adapters":    adapters,
    }, output_path)

    size_mb = output_path.stat().st_size / 1e6
    print(f"   Adapters -> {output_path.name}  ({size_mb:.1f} MB)")
    return output_path

# ── MODEL LOADING ─────────────────────────────────────────────────────────────

def load_evo2_model(model_path: Path, evo2_dir: Path):
    """Load Evo2 7B and return (model, tokenizer). Caller must chdir first."""
    sys.path.insert(0, str(evo2_dir))
    os.chdir(evo2_dir / "evo2")
    from evo2 import Evo2
    model     = Evo2(model_name="evo2_7b", local_path=str(model_path))
    tokenizer = model.tokenizer
    # vortex's load_checkpoint calls to_bfloat16_except_pr_lc() inside
    # torch.inference_mode(), so p.data becomes an inference tensor for every
    # parameter. Clone only those tensors to escape inference mode; cloning
    # outside inference_mode produces normal autograd-compatible tensors.
    for p in model.model.parameters():
        if p.data.is_inference():
            p.data = p.data.clone()
    for buf in model.model.buffers():
        if buf.is_inference():
            buf.data = buf.clone()
    return model, tokenizer

# ── ONE METHOD'S FULL RUN (training, or eval-only) ───────────────────────────

def run_one_method(args, method: str, base_data: Path, model_path: Path,
                    evo2_dir: Path, output_dir: Path):
    """
    Run a complete fine-tuning (or eval-only) pass for a single PEFT method.
    Raises on any failure -- the caller (main) decides whether subsequent
    requested methods should still run; per project decision, a failure in
    any one requested method aborts the whole invocation (fail-fast), so
    this function does not catch and continue internally.
    """
    print(f"\n{'#'*72}")
    print(f"# Method: {method.upper()}")
    print(f"{'#'*72}")

    # ── AIDRIN gate ────────────────────────────────────────────────────────
    aidrin_path = base_data / "aidrin_results" / args.aidrin_report
    if aidrin_path.exists():
        aidrin      = json.loads(aidrin_path.read_text())
        site_result = aidrin.get("sites", {}).get(args.site, {})
        if not site_result:
            print(f"[INFO] {args.site} not in AIDRIN report -- "
                  f"novel site, skipping gate")
        else:
            decision = site_result.get("gate_decision", "UNKNOWN")
            score    = site_result.get("composite_score", 0)
            if decision != "SELECTED":
                raise RuntimeError(
                    f"{args.site} was {decision} by AIDRIN (score={score:.3f}). "
                    f"Aborting."
                )
            print(f"AIDRIN gate: {args.site} SELECTED (score={score:.3f})")
    else:
        print(f"[WARN] AIDRIN report not found -- proceeding")

    # ── Held-out split ─────────────────────────────────────────────────────
    within_site_path = base_data / "within_site_held_out.json"
    held_out_mgm_id   = None
    held_out_fasta    = []

    if within_site_path.exists():
        within = json.loads(within_site_path.read_text())
        if args.site in within:
            info            = within[args.site]
            held_out_mgm_id = info["mgm_id"]
            held_out_fasta  = [Path(info["fasta_path"])]
            print(f"\nWithin-site held-out: {held_out_mgm_id} ({info['project']})")
    else:
        print("\n[WARN] within_site_held_out.json not found -- "
              "no within-site held-out split")

    ext_manifest_path = base_data / "held_out" / "held_out_manifest.json"
    if ext_manifest_path.exists():
        ext_count = len(json.loads(ext_manifest_path.read_text()))
        print(f"External held-out: {ext_count} files "
              f"(reserved for post-FedAvg likelihood ratio eval)")

    all_held_out_fastas = held_out_fasta

    # ── Device + output setup ───────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    site_output = output_dir / args.site / method
    site_output.mkdir(parents=True, exist_ok=True)

    # ── Load Evo2 ──────────────────────────────────────────────────────────
    print(f"\nLoading Evo2 7B ...")
    evo2_model, tokenizer = load_evo2_model(model_path, evo2_dir)

    # ── Inject adapters ────────────────────────────────────────────────────
    print(f"\nInjecting {method.upper()} adapters ...")
    evo2_model, _ = inject_lora(evo2_model, method=method)

    _base = evo2_model.model if hasattr(evo2_model, "model") else evo2_model
    _n_gpus = torch.cuda.device_count()
    for _dev_idx in range(_n_gpus):
        _dev = torch.device(f"cuda:{_dev_idx}")
        with torch.cuda.device(_dev):
            for _param in _base.parameters():
                if _param.device == _dev:
                    _param.data = _param.data.clone()
        torch.cuda.synchronize(_dev)

    for param in evo2_model.model.parameters():
        param.requires_grad = False
    for name, module in evo2_model.model.named_modules():
        if isinstance(module, LoRALinear):
            module.lora_A.weight.requires_grad = True
            module.lora_B.weight.requires_grad = True
        elif isinstance(module, IA3Linear):
            module.scale.requires_grad = True

    n_trainable = sum(p.numel() for p in evo2_model.model.parameters()
                       if p.requires_grad)
    print(f"   Trainable after freeze: {n_trainable:,}")

    # ── Load eval sequences ────────────────────────────────────────────────
    eval_chunks = []
    if all_held_out_fastas:
        print(f"\nLoading {args.eval_chunks} eval chunks from held-out files "
              f"(eval_chunk_size={args.eval_chunk_size}) ...")
        eval_chunks = load_eval_chunks(
            all_held_out_fastas, args.eval_chunk_size, args.eval_chunks
        )
        print(f"   Loaded {len(eval_chunks)} eval chunks")
    else:
        print("\n[WARN] No held-out files found -- skipping validation eval")

    # ── Eval-only mode ─────────────────────────────────────────────────────
    if args.eval_only:
        if not args.merged_adapter:
            raise RuntimeError("--eval-only requires --merged-adapter path")
        print(f"\nEval-only mode: loading merged adapter from {args.merged_adapter}")
        evo2_model = apply_merged_adapter(evo2_model, Path(args.merged_adapter), method)
        if eval_chunks:
            val_loss, val_ppl = evaluate(evo2_model, eval_chunks, tokenizer, device)
            result = {
                "site":            args.site,
                "method":          method,
                "merged_adapter":  args.merged_adapter,
                "eval_chunk_size": args.eval_chunk_size,
                "val_loss":        val_loss,
                "val_ppl":         val_ppl,
                "n_eval_chunks":   len(eval_chunks),
                "evaluated_at":    datetime.now(timezone.utc).isoformat(),
            }
            out_path = site_output / "fedavg_eval_result.json"
            out_path.write_text(json.dumps(result, indent=2))
            print(f"\nFedAvg eval: loss={val_loss:.4f}  PPL={val_ppl:.1f}")
            print(f"Result -> {out_path}")
        else:
            print("No eval chunks available.")
        return

    # ── Build training dataset ─────────────────────────────────────────────
    if args.data_dir:
        fasta_dir = Path(args.data_dir)
    else:
        fasta_dir = base_data / args.site / "fasta"
    print(f"\nBuilding training dataset from {fasta_dir} "
          f"(chunk_size={args.chunk_size}) ...")
    exclude = [held_out_mgm_id] if held_out_mgm_id else []
    dataset = FastaChunkDataset(
        fasta_dir     = fasta_dir,
        chunk_size    = args.chunk_size,
        max_chunks    = args.max_chunks,
        exclude_files = exclude,
    )
    dataloader = DataLoader(
        dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = True,
        num_workers = 4,
        pin_memory  = (device.type == "cuda"),
        collate_fn  = lambda b: collate_fn(b, tokenizer),
    )
    print(f"   {len(dataset):,} training chunks | "
          f"{len(dataloader):,} batches/round")

    # ── Optimizer + scheduler ──────────────────────────────────────────────
    trainable_params = [p for p in evo2_model.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_rounds * args.max_steps, eta_min=args.lr * 0.1,
    )

    # ── Training loop with convergence stopping ─────────────────────────────
    print(f"\n{'='*64}")
    print(f"  DB-FORGE Federated Fine-tuning")
    print(f"  Site:           {args.site}")
    print(f"  Method:         {method.upper()}")
    print(f"  Max rounds:     {args.max_rounds}")
    print(f"  Steps/round:    {args.max_steps}")
    print(f"  Chunk size:     {args.chunk_size}")
    print(f"  Eval chunk size:{args.eval_chunk_size}")
    print(f"  Convergence:    delta < {args.min_delta} for {args.patience} rounds")
    print(f"  Trainable:      {n_trainable:,} params")
    print(f"  Eval chunks:    {len(eval_chunks)}")
    print(f"{'='*64}\n")

    all_metrics    = []
    best_val_loss  = float("inf")
    patience_count = 0
    stop_reason    = "max_rounds"
    start_round    = args.start_round

    if start_round > 1:
        resume_path = site_output / f"round_{start_round - 1:02d}_adapters.pt"
        if resume_path.exists():
            print(f"\n   Resuming from {resume_path} (round {start_round - 1} → {start_round})")
            apply_merged_adapter(evo2_model, resume_path, method)
        else:
            print(f"   WARNING: --start-round {start_round} requested but "
                  f"{resume_path} not found — starting from round 1.")
            start_round = 1

    for round_num in range(start_round, args.max_rounds + 1):
        print(f"\n-- Round {round_num}/{args.max_rounds} ---------------------------")

        train_metrics = train_one_round(
            model=evo2_model, dataloader=dataloader, optimizer=optimizer,
            scheduler=scheduler, device=device, round_num=round_num,
            max_steps=args.max_steps, grad_accum=GRAD_ACCUM,
        )

        val_metrics = None
        train_loss  = train_metrics["avg_loss"]
        delta       = best_val_loss - train_loss

        print(f"   Train loss: {train_loss:.4f}  |  Delta: {delta:+.4f}  "
              f"({'converging' if abs(delta) < args.min_delta else 'learning'})")

        if train_loss < best_val_loss:
            best_val_loss = train_loss

        if abs(delta) < args.min_delta:
            patience_count += 1
            print(f"   [convergence] delta {abs(delta):.4f} < {args.min_delta} "
                  f"({patience_count}/{args.patience})")
            if patience_count >= args.patience:
                stop_reason = "converged"
        else:
            patience_count = 0

        all_metrics.append({**train_metrics, **(val_metrics or {})})

        adapter_path = site_output / f"round_{round_num:02d}_adapters.pt"
        export_adapters(
            model=evo2_model, output_path=adapter_path, site=args.site,
            method=method, round_num=round_num, train_metrics=train_metrics,
            val_metrics=val_metrics,
        )

        if stop_reason == "converged":
            print(f"\n   Converged after {round_num} rounds "
                  f"(delta < {args.min_delta} for {args.patience} rounds)")
            break

    log = {
        "site":             args.site,
        "method":           method,
        "max_rounds":       args.max_rounds,
        "rounds_completed": len(all_metrics),
        "stop_reason":      stop_reason,
        "max_steps":        args.max_steps,
        "chunk_size":       args.chunk_size,
        "eval_chunk_size":  args.eval_chunk_size,
        "lora_rank":        args.rank if method in ("lora", "qlora") else None,
        "learning_rate":    args.lr,
        "min_delta":        args.min_delta,
        "patience":         args.patience,
        "n_trainable":      n_trainable,
        "n_train_chunks":   len(dataset),
        "n_eval_chunks":    len(eval_chunks),
        "eval_strategy":    "within_site_held_out",
        "held_out_mgm_id":  held_out_mgm_id,
        "best_val_loss":    best_val_loss if eval_chunks else None,
        "completed_at":     datetime.now(timezone.utc).isoformat(),
        "round_metrics":    all_metrics,
        "final_train_loss": all_metrics[-1]["avg_loss"] if all_metrics else None,
        "final_val_loss":   None,
        "final_val_ppl":    None,
        "convergence_note": "Training loss used for convergence. See "
                            "evaluate_heldout.py for LLR-based generalization eval, "
                            "needed when the base model may already know this data "
                            "-- check novelty_scoring.py results first.",
    }
    log_path = site_output / "training_log.json"
    log_path.write_text(json.dumps(log, indent=2))

    print(f"\n{'='*64}")
    print(f"  Training complete ({stop_reason})  [{method.upper()}]")
    print(f"  Rounds:        {log['rounds_completed']}")
    print(f"  Final train:   {log['final_train_loss']:.4f}")
    print(f"  Adapters:      {site_output}/")
    print(f"  Log:           {log_path}")
    print(f"{'='*64}")

# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="DB-FORGE federated PEFT fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--base-data", type=str, default=None,
                     help="Base data directory (overrides DB_FORGE_BASE_DATA).")
    ap.add_argument("--model-path", type=str, default=None,
                     help="Path to evo2_7b.pt (overrides DB_FORGE_MODEL_PATH).")
    ap.add_argument("--evo2-dir", type=str, default=None,
                     help="Path to the Evo2 repo (overrides DB_FORGE_EVO2_DIR).")

    ap.add_argument("--site", required=True,
                     help="Site name (a subdirectory of --base-data). Any "
                          "name is accepted -- not restricted to a fixed list, "
                          "so new sites can be trained without code changes.")
    ap.add_argument("--data-dir", default=None, dest="data_dir",
                     help="Override the training data directory for this "
                          "site. Default: <base-data>/<site>/fasta/")
    ap.add_argument("--method", nargs="+", default=["lora"],
                     help=f"One or more PEFT methods to run, in sequence: "
                          f"{ALL_METHODS}, or 'all' for all three. "
                          f"E.g. --method lora qlora")

    ap.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS, dest="max_rounds")
    ap.add_argument("--min-delta",  type=float, default=DEFAULT_MIN_DELTA, dest="min_delta",
                     help="Convergence threshold. Validated against loss "
                          "scales of 2-6.5 in this study -- a very different "
                          "dataset may need a different value.")
    ap.add_argument("--patience",   type=int, default=DEFAULT_PATIENCE, dest="patience")
    ap.add_argument("--max-steps",  type=int, default=MAX_STEPS_PER_ROUND, dest="max_steps")
    ap.add_argument("--max-chunks", type=int, default=None, dest="max_chunks")
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, dest="chunk_size",
                     help="Training chunk size in bases. Must match your "
                          "data's sequence length regime -- see module "
                          "docstring.")
    ap.add_argument("--eval-chunk-size", type=int, default=EVAL_CHUNK_SIZE,
                     dest="eval_chunk_size",
                     help="Held-out evaluation chunk size. Independent of "
                          "--chunk-size; must be set separately.")
    ap.add_argument("--rank", type=int, default=LORA_RANK)
    ap.add_argument("--lr",   type=float, default=LEARNING_RATE)
    ap.add_argument("--eval-chunks", type=int, default=EVAL_CHUNKS, dest="eval_chunks",
                     help="Number of held-out chunks to evaluate per round")
    ap.add_argument("--aidrin-report", default="aidrin_report.json", dest="aidrin_report")
    ap.add_argument("--start-round", type=int, default=1, dest="start_round",
                     help="Resume from this round (1 = fresh start). If > 1, loads "
                          "round_{N-1:02d}_adapters.pt from the output directory and "
                          "continues training from there. Used for preemption recovery.")
    ap.add_argument("--eval-only", action="store_true", dest="eval_only",
                     help="Skip training, evaluate merged adapter on held-out data")
    ap.add_argument("--merged-adapter", default=None, dest="merged_adapter",
                     help="Path to merged adapter .pt for --eval-only mode")
    args = ap.parse_args()

    if [m.lower() for m in args.method] == ["all"]:
        methods = ALL_METHODS
    else:
        methods = [m.lower() for m in args.method]
        invalid = [m for m in methods if m not in ALL_METHODS]
        if invalid:
            print(f"ERROR: unknown method(s) {invalid}. "
                  f"Valid choices: {ALL_METHODS}, or 'all'.", file=sys.stderr)
            sys.exit(1)

    if args.eval_only and len(methods) > 1:
        print("ERROR: --eval-only expects exactly one --method "
              "(the method whose merged adapter you are evaluating).",
              file=sys.stderr)
        sys.exit(1)

    base_data  = resolve_path(args.base_data, "DB_FORGE_BASE_DATA", "base data directory")
    model_path = resolve_path(args.model_path, "DB_FORGE_MODEL_PATH", "Evo2 model path")
    evo2_dir   = resolve_path(args.evo2_dir, "DB_FORGE_EVO2_DIR", "Evo2 repo directory")
    output_dir = base_data / "peft_adapters"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"DB-FORGE Phase 4 -- Federated PEFT Fine-tuning")
    print(f"Site: {args.site}  |  Methods to run (in order): {methods}")

    # Sequential execution; fail-fast on the first method's error, per
    # project decision -- requesting multiple methods means the user wants
    # ALL of them, so a failure partway through should stop the whole run
    # rather than silently produce a partial result set.
    completed = []
    for method in methods:
        run_one_method(args, method, base_data, model_path, evo2_dir, output_dir)
        completed.append(method)

    print(f"\n{'#'*72}")
    print(f"# All requested methods completed: {completed}")
    print(f"{'#'*72}")
    print(f"\nNext: run federated_aggregate.py to merge adapters across sites")

if __name__ == "__main__":
    main()
