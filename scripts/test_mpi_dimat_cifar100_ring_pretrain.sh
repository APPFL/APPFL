#!/bin/bash
#SBATCH --job-name=dimat-c100-pt
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80GB
#SBATCH --time=08:00:00
#SBATCH --account=torch_pr_40_tandon_advanced
#SBATCH --output=appfl_dimat_c100_ring_pt_%j.out
#SBATCH --error=appfl_dimat_c100_ring_pt_%j.err

# ============================================================
# APPFL MPI DIMAT: CIFAR-100, IID, 5 agents, Ring topology
# WITH 100-epoch pre-training (matching paper setup)
#
# Paper reference (Table 2): DIMAT Ring = 67.12 ± 0.22%
# ============================================================

SCRIPT_DIR="/scratch/mp5847/src/APPFL"
cd "$SCRIPT_DIR"

# ---------- Activate conda environment ----------
eval "$(conda shell.bash hook)"
conda activate /scratch/mp5847/conda_environments/appfl

set -euo pipefail

# ---------- Configuration ----------
NUM_CLIENTS=5
NUM_PROCS=$((NUM_CLIENTS + 1))  # 1 server + N clients
SERVER_CONFIG="examples/resources/configs/cifar100/server_dimat_ring.yaml"
CLIENT_CONFIG="examples/resources/configs/cifar100/client_dimat_ring.yaml"
PRETRAIN_EPOCHS=100

# ---------- Environment verification ----------
verify_env() {
    echo "=== Verifying environment ==="
    python -c "from mpi4py import MPI; print('mpi4py OK')"
    python -c "import appfl; print(f'appfl {appfl.__version__} OK')"
    python -c "from appfl.algorithm.aggregator import DIMATaggregator; print('DIMATaggregator OK')"
    python -c "from appfl.algorithm.trainer import DIMATTrainer; print('DIMATTrainer OK')"
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}')"
    echo "=== Environment ready ==="
    echo ""
}

# ---------- Pre-download CIFAR-100 ----------
download_data() {
    echo "=== Downloading CIFAR-100 data (if needed) ==="
    cd "$SCRIPT_DIR/examples"
    python -c "
import torchvision, os
d = os.path.join(os.getcwd(), 'datasets', 'RawData')
torchvision.datasets.CIFAR100(d, download=True, train=True)
torchvision.datasets.CIFAR100(d, download=True, train=False)
print('CIFAR-100 data ready at', d)
"
    cd "$SCRIPT_DIR"
    echo ""
}

# ---------- Run the experiment ----------
run_experiment() {
    echo "=== Running APPFL MPI DIMAT: CIFAR-100 IID Ring (with pre-training) ==="
    echo "  Model           : ResNet20 (w=8, channels 128/256/512, 100 classes)"
    echo "  Server config   : $SERVER_CONFIG"
    echo "  Client config   : $CLIENT_CONFIG"
    echo "  MPI processes   : $NUM_PROCS (1 server + $NUM_CLIENTS clients)"
    echo "  Partition       : IID"
    echo "  Topology        : Ring (each agent merges with 2 neighbors)"
    echo "  Pre-train       : $PRETRAIN_EPOCHS epochs (local, independent)"
    echo "  Merge-train     : 100 rounds x 2 local epochs"
    echo "  Optimizer       : Adam lr=0.001"
    echo "  Batch size      : 100"
    echo ""

    cd "$SCRIPT_DIR/examples"

    mpirun --oversubscribe -n "$NUM_PROCS" python mpi/run_mpi_dimat_pretrain.py \
        --server_config "../$SERVER_CONFIG" \
        --client_config "../$CLIENT_CONFIG" \
        --pretrain_epochs "$PRETRAIN_EPOCHS"

    cd "$SCRIPT_DIR"

    echo ""
    echo "=== Experiment finished ==="
    echo "Check examples/output/ for logs and results."
}

# ---------- Main ----------
main() {
    echo "============================================"
    echo " DIMAT: CIFAR-100, IID, 5 agents, Ring"
    echo " Pre-training: $PRETRAIN_EPOCHS epochs"
    echo " Paper target: 67.12 ± 0.22%"
    echo " $(date)"
    echo "============================================"
    echo ""

    verify_env
    download_data
    run_experiment
}

main
