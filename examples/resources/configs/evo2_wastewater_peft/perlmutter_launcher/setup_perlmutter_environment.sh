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
module load cudatoolkit/12.4
module load gcc-native/12.3

export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export CUDAHOSTCXX=/usr/bin/g++-12
export PIP_CACHE_DIR TMPDIR TRITON_CACHE_DIR

conda create -y -p "${ENV_DIR}" python=3.12
conda activate "${ENV_DIR}"

# torch + torchvision together, same cu128 index, before appfl: appfl pulls
# in an unpinned torchvision that otherwise mismatches torch's CUDA version.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

pip install "appfl>=1.12.0"

# flash-attn needs its own --no-build-isolation build; evo2/vtx don't pull
# in a working one. MAX_JOBS caps nvcc parallelism so it doesn't OOM.
export MAX_JOBS=2
pip install flash-attn --no-build-isolation

pip install evo2

python3 -c "
import torch, evo2, flash_attn
from appfl.agent.client import ClientAgent
print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
print('Flash Attention:', flash_attn.__version__)
print('evo2 OK, appfl.agent.client OK')
"

echo "Env ready at ${ENV_DIR}"
