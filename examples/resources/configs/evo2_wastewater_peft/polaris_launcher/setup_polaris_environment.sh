#!/bin/bash

set -euo pipefail

if [ -z "${1:-}" ]; then
    echo "Usage: bash setup_polaris_environment.sh <base_dir>"
    exit 1
fi

BASE_DIR="$1"
ENV_DIR="${BASE_DIR}/envs/db_forge_evo2"
PIP_CACHE_DIR="${BASE_DIR}/.pip_cache"
TMPDIR="${BASE_DIR}/.tmp"
TRITON_CACHE_DIR="${BASE_DIR}/.triton_cache"

if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Run this on a Polaris compute node."
    exit 1
fi

mkdir -p "${PIP_CACHE_DIR}" "${TMPDIR}" "${TRITON_CACHE_DIR}"

module use /soft/modulefiles
module load conda/2025-09-28
module load cudatoolkit-standalone/12.6.0
module load gcc-native/14

export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
export CUDAHOSTCXX="$(command -v g++)"
export PIP_CACHE_DIR
export TMPDIR
export TRITON_CACHE_DIR

conda create -y -p "${ENV_DIR}" python=3.12
conda activate "${ENV_DIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pip install -r "${SCRIPT_DIR}/requirements.txt"

pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126

CONSTRAINTS_FILE="${TMPDIR}/torch-constraints.txt"
echo "torch==2.7.1" > "${CONSTRAINTS_FILE}"

EVO2_DIR="${BASE_DIR}/evo2"
if [ ! -d "${EVO2_DIR}" ]; then
    git clone https://github.com/ArcInstitute/evo2.git "${EVO2_DIR}"
fi
pip install "${EVO2_DIR}" -c "${CONSTRAINTS_FILE}"

export MAX_JOBS=4
pip install flash-attn --no-build-isolation -c "${CONSTRAINTS_FILE}"

APPFL_REPO_ROOT="$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel)"
pip install "${APPFL_REPO_ROOT}" -c "${CONSTRAINTS_FILE}"

python3 -c "
import torch, flash_attn, appfl
print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
print('Flash Attention:', flash_attn.__version__)
print('APPFL:', appfl.__version__)
try:
    import transformer_engine
    print('Transformer Engine:', transformer_engine.__version__)
except ImportError:
    print('Transformer Engine: not installed (optional, FP8-only; evo2_7b falls back to bf16)')
"

echo "Env ready at ${ENV_DIR}"
