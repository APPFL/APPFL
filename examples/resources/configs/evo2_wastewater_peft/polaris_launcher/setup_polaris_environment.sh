#!/bin/bash
# Usage: bash setup_polaris_environment.sh <base_dir>
#   e.g. bash setup_polaris_environment.sh /lus/grand/projects/<your_project>/db-forge
# Run on a Polaris compute node (interactive job), not a login node.

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
module load gcc-native/12.3

export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export CUDAHOSTCXX=/usr/bin/g++-12
export PIP_CACHE_DIR
export TMPDIR
export TRITON_CACHE_DIR

conda create -y -p "${ENV_DIR}" python=3.12
conda activate "${ENV_DIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pip install -r "${SCRIPT_DIR}/requirements.txt"

pip install flash-attn      --no-build-isolation
pip install mamba-ssm       --no-build-isolation
pip install causal-conv1d   --no-build-isolation

EVO2_DIR="${BASE_DIR}/evo2"
if [ ! -d "${EVO2_DIR}" ]; then
    git clone https://github.com/ArcInstitute/evo2.git "${EVO2_DIR}"
fi
pip install -e "${EVO2_DIR}"

python3 -c "
import torch, transformer_engine, flash_attn
print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
print('Transformer Engine:', transformer_engine.__version__)
print('Flash Attention:', flash_attn.__version__)
"

echo "Env ready at ${ENV_DIR}"
