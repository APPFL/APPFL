#!/usr/bin/env bash
set -euo pipefail

APPFL_ROOT="${APPFL_ROOT:-/Users/johnzhouyang.wu/Documents/APPFL}"
PYTHON="${PYTHON:-/opt/homebrew/Caskroom/miniconda/base/envs/adko/bin/python}"
REF_ROOT="${REF_ROOT:-$APPFL_ROOT/.external/adko_ref}"
OUT_ROOT="${OUT_ROOT:-$APPFL_ROOT/results/appfl_suzuki_main_llmoff}"

cd "$APPFL_ROOT"
export PYTHONPATH="$APPFL_ROOT/src:."

COMMON_ARGS=(
  --warmup-dir "$REF_ROOT/scientific_discovery/results/warmup"
)
if [[ -n "${SEEDS:-}" ]]; then
  COMMON_ARGS+=(--seeds "$SEEDS")
fi
if [[ -n "${PARALLEL:-}" ]]; then
  COMMON_ARGS+=(--parallel "$PARALLEL")
fi
if [[ -n "${P_NOISE:-}" ]]; then
  COMMON_ARGS+=(--p-noise "$P_NOISE")
fi

"$PYTHON" examples/decentralized/suzuki/run_suzuki_appfl.py \
  --config "$REF_ROOT/scientific_discovery/experiments/main_iid_llmoff.json" \
  --out-dir "$OUT_ROOT/iid" \
  "${COMMON_ARGS[@]}"

"$PYTHON" examples/decentralized/suzuki/run_suzuki_appfl.py \
  --config "$REF_ROOT/scientific_discovery/experiments/main_noniid_llmoff.json" \
  --out-dir "$OUT_ROOT/het" \
  "${COMMON_ARGS[@]}"
