#!/usr/bin/env bash
set -euo pipefail

APPFL_ROOT="${APPFL_ROOT:-/Users/johnzhouyang.wu/Documents/APPFL}"
PYTHON="${PYTHON:-/opt/homebrew/Caskroom/miniconda/base/envs/adko/bin/python}"
REF_ROOT="${REF_ROOT:-$APPFL_ROOT/.external/adko_ref}"
OUT_ROOT="${OUT_ROOT:-$APPFL_ROOT/results/appfl_suzuki_smoke}"
SEEDS="${SEEDS:-1}"
PARALLEL="${PARALLEL:-1}"
P_NOISE="${P_NOISE:-0}"

cd "$APPFL_ROOT"
export PYTHONPATH="$APPFL_ROOT/src:."

"$PYTHON" examples/decentralized/suzuki/run_suzuki_appfl.py \
  --config "$REF_ROOT/scientific_discovery/experiments/main_iid_llmoff.json" \
  --warmup-dir "$REF_ROOT/scientific_discovery/results/warmup" \
  --out-dir "$OUT_ROOT/iid" \
  --seeds "$SEEDS" \
  --parallel "$PARALLEL" \
  --p-noise "$P_NOISE"

"$PYTHON" examples/decentralized/suzuki/run_suzuki_appfl.py \
  --config "$REF_ROOT/scientific_discovery/experiments/main_noniid_llmoff.json" \
  --warmup-dir "$REF_ROOT/scientific_discovery/results/warmup" \
  --out-dir "$OUT_ROOT/het" \
  --seeds "$SEEDS" \
  --parallel "$PARALLEL" \
  --p-noise "$P_NOISE"
