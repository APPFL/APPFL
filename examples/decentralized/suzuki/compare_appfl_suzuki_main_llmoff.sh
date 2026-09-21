#!/usr/bin/env bash
set -euo pipefail

APPFL_ROOT="${APPFL_ROOT:-/Users/johnzhouyang.wu/Documents/APPFL}"
PYTHON="${PYTHON:-/opt/homebrew/Caskroom/miniconda/base/envs/adko/bin/python}"
REF_ROOT="${REF_ROOT:-$APPFL_ROOT/.external/adko_ref}"
OUT_ROOT="${OUT_ROOT:-$APPFL_ROOT/results/appfl_suzuki_main_llmoff}"
FIG_ROOT="${FIG_ROOT:-$OUT_ROOT/figures}"
REPORT_ROOT="${REPORT_ROOT:-$OUT_ROOT/reports}"
N_BOOT="${N_BOOT:-5000}"
CI="${CI:-95}"

cd "$APPFL_ROOT"
export PYTHONPATH="$APPFL_ROOT/src:."

mkdir -p "$FIG_ROOT" "$REPORT_ROOT"

COMMON_ARGS=(
  --n-boot "$N_BOOT"
  --ci "$CI"
)
if [[ "${STRICT_STEPS:-0}" == "1" ]]; then
  COMMON_ARGS+=(--strict-steps)
fi

"$PYTHON" examples/decentralized/suzuki/compare_to_reference.py \
  --appfl "$OUT_ROOT/iid" \
  --reference "$REF_ROOT/scientific_discovery/results/main/b2l1g32s0p5t50tb40a0p01pn0p04365_llmoff" \
  --arm IID \
  --plot "$FIG_ROOT/iid_llmoff.png" \
  --report-json "$REPORT_ROOT/iid_llmoff_comparison.json" \
  "${COMMON_ARGS[@]}"

"$PYTHON" examples/decentralized/suzuki/compare_to_reference.py \
  --appfl "$OUT_ROOT/het" \
  --reference "$REF_ROOT/scientific_discovery/results/main/b2l4g32s0p5t50tb40a0p01pn0p04365_llmoff" \
  --arm HET \
  --plot "$FIG_ROOT/het_llmoff.png" \
  --report-json "$REPORT_ROOT/het_llmoff_comparison.json" \
  "${COMMON_ARGS[@]}"
