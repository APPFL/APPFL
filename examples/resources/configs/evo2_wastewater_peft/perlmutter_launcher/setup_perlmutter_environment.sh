#!/bin/bash
# Usage: bash setup_perlmutter_environment.sh <base_dir>
#   e.g. bash setup_perlmutter_environment.sh /pscratch/sd/<u>/<username>/dbforge/evo2_metagenomics
# Run on a Perlmutter login node (no GPU needed to build the env itself).

set -euo pipefail

if [ -z "${1:-}" ]; then
    echo "Usage: bash setup_perlmutter_environment.sh <base_dir>"
    exit 1
fi

BASE_DIR="$1"
ENV_DIR="${BASE_DIR}/envs/evo2_fl_env312"
PIP_CACHE_DIR="${BASE_DIR}/.pip_cache"
TMPDIR="${BASE_DIR}/.tmp"
TRITON_CACHE_DIR="${BASE_DIR}/.triton_cache"

mkdir -p "${PIP_CACHE_DIR}" "${TMPDIR}" "${TRITON_CACHE_DIR}"

module load conda
export PIP_CACHE_DIR TMPDIR TRITON_CACHE_DIR

conda create -y -p "${ENV_DIR}" python=3.12
conda activate "${ENV_DIR}"

# PyTorch with the CUDA 12.8 build matching Perlmutter's A100 driver.
# Install this BEFORE appfl: appfl's own requirement is just "torch" with no
# version pin, so once this is installed, installing appfl afterward won't
# touch it or swap in a different CUDA build.
pip install torch --index-url https://download.pytorch.org/whl/cu128

# appfl >= 1.12.0 — this is the first tagged release with the gRPC polling
# feature, needed for long-running FL rounds behind a public TLS relay
# (e.g. APPFLx). Pulls in grpc, pandas, numpy, and most other deps.
pip install "appfl>=1.12.0"

# Evo2 — pulls in biopython, huggingface_hub, and vtx (which itself installs
# flash-attn, the actual GPU kernels for Evo2's architecture).
pip install evo2

python3 -c "
import torch, appfl, evo2
print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
print('evo2 OK')
"

echo "Env ready at ${ENV_DIR}"
