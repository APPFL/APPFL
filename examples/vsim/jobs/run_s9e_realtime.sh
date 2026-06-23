#!/bin/bash -l
#PBS -N vsim-s9e-real
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

OUT=./vsim_logs/s9e
mkdir -p "$OUT"

CFG=./vsim/config_vsim_cifar_async.yaml
CLI=./resources/configs/cifar10/client_1.yaml

echo "======== S9e: 500 rounds, async CUDA, real training time (no base_step_time) ========"
python vsim/run_vsim.py \
    --server_config "$CFG" --client_config "$CLI" \
    --num_clients 10 --num_global_epochs 500 \
    --num_local_steps 20 --max_concurrency 4 \
    --device cuda --seed 42 --verify \
    --partition dirichlet_noniid --alpha 1.0 --num_classes 10 \
    --eval_every 10 \
    2>&1 | tee "${OUT}/run_s9e_realtime_500r.log"
echo "======== DONE ========"
