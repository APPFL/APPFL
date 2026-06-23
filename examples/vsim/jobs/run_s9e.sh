#!/bin/bash -l
#PBS -N vsim-s9e
#PBS -A PPFL_FM
#PBS -q preemptable
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=1:00:00
#PBS -l filesystems=home:eagle
#PBS -r y
#PBS -k doe
#PBS -j oe

set -euo pipefail
cd "${PBS_O_WORKDIR:-$PWD}"

module use /soft/modulefiles
module load conda
conda activate base

export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export ftp_proxy="http://proxy.alcf.anl.gov:3128"
export PYTHONPATH=/home/sungminkang/ANL/virtual_sim/APPFL/src:${PYTHONPATH:-}
export OPENBLAS_NUM_THREADS=16

WORK=/home/sungminkang/ANL/virtual_sim/APPFL/examples
cd "$WORK"

CFG=./vsim/config_vsim_cifar_async.yaml
CLI=./resources/configs/cifar10/client_1.yaml
OUT=./vsim_logs/s9e
mkdir -p "$OUT"

run_exp() {
    local tag=$1 nc=$2 epochs=$3 steps=$4 dir_alpha=${5:-1.0}
    echo "======== RUN $tag: N=$nc rounds=$epochs steps=$steps alpha=$dir_alpha ========"
    python vsim/run_vsim.py \
        --server_config "$CFG" --client_config "$CLI" \
        --num_clients "$nc" --num_global_epochs "$epochs" \
        --num_local_steps "$steps" --max_concurrency 4 \
        --device cuda --seed 42 --verify \
        --partition dirichlet_noniid --alpha "$dir_alpha" \
        --base_step_time 0.01 --eval_every 10 \
        2>&1 | tee "${OUT}/run_${tag}.log"
    echo "======== DONE $tag ========"
    echo ""
}

# EXP-1: rounds sweep (N=10, K=4, steps=20)
run_exp A 10 10 20
run_exp B 10 20 20
run_exp C 10 30 20

# EXP-3: steps sweep (N=10, K=4, rounds=20)
run_exp D 10 20 10
# B is reused for steps=20
run_exp E 10 20 30

# EXP-2: N sweep (K=4, steps=20, rounds=3*N)
run_exp F 5  15 20
# C is reused for N=10,rounds=30
run_exp G 20 60 20 2.0

echo "All S9e runs complete."
