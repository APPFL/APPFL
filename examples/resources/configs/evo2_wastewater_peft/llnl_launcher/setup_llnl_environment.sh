#!/bin/bash
# Usage: bash setup_llnl_environment.sh <base_dir>
#   e.g. bash setup_llnl_environment.sh /p/lustre1/<username>/dbforge/evo2_metagenomics

set -euo pipefail

if [ -z "${1:-}" ]; then
    echo "Usage: bash setup_llnl_environment.sh <base_dir>"
    exit 1
fi

BASE_DIR="$1"
ENV_DIR="${BASE_DIR}/envs/evo2_fl_env312"
PIP_CACHE_DIR="${BASE_DIR}/.pip_cache"
TMPDIR="${BASE_DIR}/.tmp"
TRITON_CACHE_DIR="${BASE_DIR}/.triton_cache"

mkdir -p "${PIP_CACHE_DIR}" "${TMPDIR}" "${TRITON_CACHE_DIR}"

module load conda
module load cuda/12.6
module load gcc/12

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

# pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
# CONSTRAINTS_FILE="${TMPDIR}/torch-constraints.txt"
# echo "torch==2.7.1" > "${CONSTRAINTS_FILE}"
#
# EVO2_DIR="${BASE_DIR}/evo2"
# if [ ! -d "${EVO2_DIR}" ]; then
#     git clone https://github.com/ArcInstitute/evo2.git "${EVO2_DIR}"
# fi
# pip install "${EVO2_DIR}" -c "${CONSTRAINTS_FILE}"
#
# export MAX_JOBS=4
# pip install flash-attn==2.8.3.post2 --no-build-isolation -c "${CONSTRAINTS_FILE}"

APPFL_REPO_ROOT="$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel)"
pip install "${APPFL_REPO_ROOT}"

python3 -c "
import torch, appfl
print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
print('APPFL:', appfl.__version__)
"

echo "Env ready at ${ENV_DIR}"
