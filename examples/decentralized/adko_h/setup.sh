#!/usr/bin/env bash
# One-time setup for ADKO-H, on a LOGIN node (compute nodes have no internet).
# Creates a small environment, downloads the molecule pool, and pre-builds the two
# caches a compute node would otherwise rebuild in every job.
#
#   export ADKO_H=/path/to/this/APPFL          # the checkout holding src/appfl
#   export ADKO_DATA=$ADKO_H/../adko_h_data    # anywhere writable, not in the repo
#   bash examples/decentralized/adko_h/setup.sh
#
# Then, in any shell that runs ADKO-H:
#   conda activate "${ADKO_ENV:-adko_h}"
#   export PYTHONPATH=$ADKO_H/src:$PYTHONPATH
#   export ADKO_PMO_POOL=$ADKO_DATA/zinc250k.smi
#
# The environment is deliberately tiny: ADKO-H itself needs numpy and scipy, the molecule
# problem needs rdkit, and the planner needs the openai client to talk to vLLM. No torch,
# so this takes a minute rather than twenty, and APPFL is used from PYTHONPATH rather than
# installed, which keeps it out of the way of any other environment on the cluster.
set -eu

ENV_NAME=${ADKO_ENV:-adko_h}
PYVER=${ADKO_PYTHON:-3.12}
ADKO_H=${ADKO_H:?set ADKO_H to the APPFL checkout that contains src/appfl}
ADKO_DATA=${ADKO_DATA:-$ADKO_H/../adko_h_data}
POOL_LIMIT=${POOL_LIMIT:-}

echo "== environment $ENV_NAME (python $PYVER)"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" "python=$PYVER"
fi
conda run -n "$ENV_NAME" pip install --upgrade \
  "numpy>=1.26" "scipy>=1.11" "rdkit>=2024.3" "openai>=1.40" omegaconf  # adko_v2 imports it

echo "== molecule pool into $ADKO_DATA"
mkdir -p "$ADKO_DATA"
export PYTHONPATH="$ADKO_H/src:${PYTHONPATH:-}"
PKG=appfl.decentralized.algorithm.adko_h.problems.pmo_oracles
conda run -n "$ENV_NAME" --no-capture-output \
  python -I "$ADKO_H/src/${PKG//.//}/fetch_pool.py" "$ADKO_DATA"

POOL=$ADKO_DATA/zinc250k.smi
echo "== fingerprint and embedding caches (a few minutes, once)"
conda run -n "$ENV_NAME" --no-capture-output python -W ignore -c "
import os, sys
os.environ.setdefault('ADKO_PMO_POOL', '$POOL')
from appfl.decentralized.algorithm.adko_h.problems.pmo import PMO
limit = int('${POOL_LIMIT:-0}') or None
p = PMO('qed', pool_path='$POOL', limit=limit, verbose=True)
print('pool ready:', p.n, 'molecules, descriptor dim', p.descriptor_dim())
"

echo "== self-check: the test suite, offline"
cd "$ADKO_H"
ADKO_PMO_POOL=$POOL conda run -n "$ENV_NAME" --no-capture-output \
  python -W ignore -m unittest discover -s tests/decentralized -t . 2>&1 | tail -4

cat <<EOF

Ready. In every shell that runs ADKO-H:
  conda activate $ENV_NAME
  export PYTHONPATH=$ADKO_H/src:\$PYTHONPATH
  export ADKO_PMO_POOL=$POOL
  export ADKO_RESULTS=\${ADKO_RESULTS:-$ADKO_DATA/results}
EOF
