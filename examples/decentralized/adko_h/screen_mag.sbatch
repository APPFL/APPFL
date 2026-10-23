#!/bin/bash
#SBATCH --job-name=adko_h_mag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
# ADKO-H arms planned through the AmSC gateway (MAG) instead of a served model. CPU only.
#
# Everything runs in ONE job on ONE node, on purpose: MAG's limits are per user (20 requests
# in flight, 120 a minute) and the slot files that respect them are local, so arms spread
# over several nodes would each think they had the whole allowance. Within this job, PAR runs
# go at once and share the node's slots.
#
# Money is explicit. MAG_BUDGET is required and caps the ledger every run here writes to; a
# run that reaches it stops with BudgetSpent rather than continuing on random designs. The
# gateway's own figures are printed before and after.
#
#   export ADKO_H=/path/to/APPFL ADKO_DATA=/path/to/adko_h_data
#   export AMSC_TOKEN_FILE=/path/to/token   # chmod 600; never inside a repo
#   MODEL=claude-sonnet-5-5-medium@amazon MAG_BUDGET=25 \
#     ARMS="adkoh" SEEDS="42 43" sbatch -p <cpu-partition> \
#     examples/decentralized/adko_h/screen_mag.sbatch
#
# Variables: MODEL (required), MAG_BUDGET (required), ARMS, SEEDS, ORACLE, ROUNDS, AGENTS,
# PAR, CAMPAIGN_MIN, CAMPAIGN_MAX, plus the topology/pool ones screen.sbatch takes.
set -eu

MODEL=${MODEL:?set MODEL to a MAG model id, e.g. claude-sonnet-5-5-medium@amazon}
: "${MAG_BUDGET:?set MAG_BUDGET to the dollar cap for this job}"
ARMS=${ARMS:-adkoh}
SEEDS=${SEEDS:-42 43 44 45 46 47 48 49}
ORACLE=${ORACLE:-zaleplon_mpo}
ROUNDS=${ROUNDS:-1000}
AGENTS=${AGENTS:-10}
PAR=${PAR:-4}
CAMPAIGN_MIN=${CAMPAIGN_MIN:-20}
CAMPAIGN_MAX=${CAMPAIGN_MAX:-60}
TOPOLOGY=${TOPOLOGY:-random_regular}
DEGREE=${DEGREE:-4}
POOL_SIZE=${POOL_SIZE:-256}
HIT=${HIT:-0.4}
ADKO_H=${ADKO_H:?set ADKO_H to the APPFL checkout that contains src/appfl}
DATA=${ADKO_DATA:?set ADKO_DATA to a writable directory}
RESULTS=${ADKO_RESULTS:-$DATA/results}

export PYTHONPATH="$ADKO_H/src:${PYTHONPATH:-}"
export ADKO_PMO_POOL=${ADKO_PMO_POOL:-$DATA/zinc250k.smi}
export MAG_LEDGER=${MAG_LEDGER:-$DATA/mag_ledger.txt}
export MAG_SLOTS=${MAG_SLOTS:-18} MAG_GAP=${MAG_GAP:-10}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONWARNINGS=ignore
PY="conda run -n ${ADKO_ENV:-adko_h} --no-capture-output python -W ignore"
MAG="$PY -m appfl.decentralized.algorithm.adko_h.llm.mag"

cd "$ADKO_H"
echo "== gateway reachable from $(hostname -s)?"
$MAG ping --model "$MODEL" || { echo "MAG unreachable from this node; see AGENTS.md"; exit 1; }
echo "== gateway spend before"; $MAG spend
echo "== ledger before: $(awk '{s+=$1} END {printf "$%.2f", s+0}' "$MAG_LEDGER" 2>/dev/null || echo '$0')  cap \$$MAG_BUDGET"

label() { echo "$1-${MODEL%%@*}"; }
run_one() {  # arm seed
  $PY -m appfl.decentralized.algorithm.adko_h.run \
    --problem pmo --oracle "$ORACLE" --arm "$1" --planner mag --llm-model "$MODEL" \
    --label "$(label "$1")" --rounds "$ROUNDS" --agents "$AGENTS" --seeds "$2" \
    --topology "$TOPOLOGY" --degree "$DEGREE" --pool-size "$POOL_SIZE" \
    --campaign-min "$CAMPAIGN_MIN" --campaign-max "$CAMPAIGN_MAX" \
    --hit-threshold "$HIT" --start-points 20 --warmup 3 \
    --out "$RESULTS/$ORACLE" > "$RESULTS/$ORACLE.$(label "$1").s$2.log" 2>&1
}

mkdir -p "$RESULTS"
status=0
for seed in $SEEDS; do
  for arm in $ARMS; do
    while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 5; done
    out="$RESULTS/$ORACLE/pmo-$ORACLE/$(label "$arm")/seed$seed/run.json"
    [ -s "$out" ] && { echo "skip $arm seed $seed (done)"; continue; }
    ( run_one "$arm" "$seed" && echo "done $arm seed $seed" \
      || echo "FAILED $arm seed $seed: $(tail -n 1 "$RESULTS/$ORACLE.$(label "$arm").s$seed.log")" ) &
    sleep 2
  done
done
wait || status=$?

echo "== ledger after: $(awk '{s+=$1} END {printf "$%.2f", s+0}' "$MAG_LEDGER")  cap \$$MAG_BUDGET"
echo "== gateway spend after"; $MAG spend
exit $status
