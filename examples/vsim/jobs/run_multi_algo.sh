#!/bin/bash -l
#PBS -N vsim-multi-algo
#PBS -A PPFL_FM
#PBS -q preemptable
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=3:00:00
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

CLI=./resources/configs/cifar10/client_1.yaml
COMMON="--num_clients 10 --num_global_epochs 500 --num_local_steps 20 --max_concurrency 4 --device cuda --seed 42 --verify --partition dirichlet_noniid --alpha 1.0 --num_classes 10 --eval_every 10"

# --- Exp 1: FedBuff + CNN (same as S9e but FedBuff, K=3) ---
OUT1=./vsim_logs/fedbuff_cnn
mkdir -p "$OUT1"
echo "======== Exp1: FedBuff + CNN, 500 rounds ========"
python vsim/run_vsim.py \
    --server_config ./vsim/config_vsim_cifar_fedbuff.yaml \
    --client_config "$CLI" $COMMON \
    2>&1 | tee "${OUT1}/run_fedbuff_cnn_500r.log"
echo "======== Exp1 DONE ========"

# --- Exp 2: FedAsync + ResNet-18 ---
OUT2=./vsim_logs/fedasync_resnet
mkdir -p "$OUT2"
echo "======== Exp2: FedAsync + ResNet-18, 500 rounds ========"
python vsim/run_vsim.py \
    --server_config ./vsim/config_vsim_cifar_resnet_async.yaml \
    --client_config "$CLI" $COMMON \
    2>&1 | tee "${OUT2}/run_fedasync_resnet_500r.log"
echo "======== Exp2 DONE ========"

# --- Exp 3: FedBuff + ResNet-18 ---
OUT3=./vsim_logs/fedbuff_resnet
mkdir -p "$OUT3"
echo "======== Exp3: FedBuff + ResNet-18, 500 rounds ========"
python vsim/run_vsim.py \
    --server_config ./vsim/config_vsim_cifar_resnet_fedbuff.yaml \
    --client_config "$CLI" $COMMON \
    2>&1 | tee "${OUT3}/run_fedbuff_resnet_500r.log"
echo "======== Exp3 DONE ========"

echo "All experiments finished."
