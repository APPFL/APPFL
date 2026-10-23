#!/bin/bash
#SBATCH --job-name=adko_h_screen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --array=0-7
# One ADKO-H arm over the seeds of this array task. The problem is CPU-only -- the oracle is
# rdkit, not a simulation -- so the no-LLM arms need no GPU at all, and the LLM arms need
# only the one GPU that serve.sbatch already holds.
#
# Each array task is one seed, so a failed seed is re-run on its own and the block is half
# the work. Submit one array per arm:
#
#   export ADKO_H=/path/to/APPFL ADKO_DATA=/path/to/adko_h_data
#   export ADKO_PMO_POOL=$ADKO_DATA/zinc250k.smi ADKO_RESULTS=$ADKO_DATA/results
#   for ARM in adko adko-div random-campaign; do
#     ARM=$ARM sbatch -p <cpu-partition> examples/decentralized/adko_h/screen.sbatch
#   done
#   # after serve.sbatch is up:
#   for ARM in adkoh adkoh-noclaims adkoh-noterritory adkoh-nofootprint; do
#     ARM=$ARM sbatch -p <cpu-partition> examples/decentralized/adko_h/screen.sbatch
#   done
#
# Variables: ARM, ORACLE, ROUNDS, AGENTS, SEED0, DEGREE, TOPOLOGY, POOL_SIZE, HIT,
# CAMPAIGN_MIN, CAMPAIGN_MAX.
set -eu

ARM=${ARM:-adko}
ORACLE=${ORACLE:-zaleplon_mpo}
ROUNDS=${ROUNDS:-1000}          # ROUNDS * AGENTS = PMO's 10,000-call budget
AGENTS=${AGENTS:-10}
SEED0=${SEED0:-42}
TOPOLOGY=${TOPOLOGY:-random_regular}
DEGREE=${DEGREE:-4}
POOL_SIZE=${POOL_SIZE:-256}
HIT=${HIT:-0.4}
CAMPAIGN_MIN=${CAMPAIGN_MIN:-20}  # campaigns of 20-60 at the full budget (README); run.py's
CAMPAIGN_MAX=${CAMPAIGN_MAX:-60}  # own default of 2-10 would mean ~4x the model calls
ADKO_H=${ADKO_H:?set ADKO_H to the APPFL checkout that contains src/appfl}
DATA=${ADKO_DATA:?set ADKO_DATA to a writable directory}
RESULTS=${ADKO_RESULTS:-$DATA/results}
SEED=$((SEED0 + ${SLURM_ARRAY_TASK_ID:-0}))

export PYTHONPATH="$ADKO_H/src:${PYTHONPATH:-}"
export ADKO_PMO_POOL=${ADKO_PMO_POOL:-$DATA/zinc250k.smi}
# One BLAS thread per process: the GPs here are small, and threads only fight each other.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONWARNINGS=ignore

PLANNER=none
case "$ARM" in
  adko|adko-div) PLANNER=none ;;
  random-campaign) PLANNER=random ;;
  adkoh*) PLANNER=vllm ;;
esac

if [ "$PLANNER" = vllm ]; then
  ENDPOINT_FILE=${ADKO_LLM_ENDPOINT_FILE:-$DATA/llm_endpoint.txt}
  for _ in $(seq 1 180); do
    [ -s "$ENDPOINT_FILE" ] && break
    sleep 10
  done
  [ -s "$ENDPOINT_FILE" ] || { echo "no LLM endpoint at $ENDPOINT_FILE; start serve.sbatch"; exit 1; }
  export ADKO_LLM_BASE_URL=$(cat "$ENDPOINT_FILE")
  echo "planner endpoint $ADKO_LLM_BASE_URL"
fi

echo "arm=$ARM oracle=$ORACLE seed=$SEED rounds=$ROUNDS agents=$AGENTS planner=$PLANNER"
cd "$ADKO_H"
exec conda run -n "${ADKO_ENV:-adko_h}" --no-capture-output \
  python -W ignore -m appfl.decentralized.algorithm.adko_h.run \
    --problem pmo --oracle "$ORACLE" --arm "$ARM" --planner "$PLANNER" \
    --rounds "$ROUNDS" --agents "$AGENTS" --seeds "$SEED" \
    --topology "$TOPOLOGY" --degree "$DEGREE" --pool-size "$POOL_SIZE" \
    --hit-threshold "$HIT" --start-points 20 --warmup 3 \
    --campaign-min "$CAMPAIGN_MIN" --campaign-max "$CAMPAIGN_MAX" \
    --llm-model "${ADKO_LLM_MODEL:-Qwen/Qwen3.8-27B-FP8}" \
    --llm-thinking "${ADKO_LLM_THINKING:-low}" \
    --out "$RESULTS/$ORACLE"
