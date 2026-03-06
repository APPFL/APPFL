#!/bin/bash
#SBATCH --job-name=dimat-5c
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80GB
#SBATCH --time=04:00:00
#SBATCH --account=torch_pr_40_tandon_advanced
#SBATCH --output=appfl_dimat_5c_%j.out
#SBATCH --error=appfl_dimat_5c_%j.err

# ============================================================
# APPFL MPI DIMAT Paper Replication (GPU)
#
# CIFAR-10, 5 clients, 2 classes/client (non-IID),
# ResNet20 (w=8), fully connected topology.
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
SERVER_CONFIG="examples/resources/configs/cifar10/server_dimat_5c.yaml"
CLIENT_CONFIG="examples/resources/configs/cifar10/client_dimat_5c.yaml"

# ---------- Environment verification ----------
verify_env() {
    echo "=== Verifying environment ==="
    python -c "from mpi4py import MPI; print('mpi4py OK')"
    python -c "import appfl; print(f'appfl {appfl.__version__} OK')"
    python -c "from appfl.algorithm.aggregator import DIMATaggregator; print('DIMATaggregator OK')"
    python -c "from appfl.algorithm.trainer import DIMATTrainer; print('DIMATTrainer OK')"
    python -c "import scipy; print('scipy OK')"
    python -c "import networkx; print('networkx OK')"
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}')"
    echo "mpirun: $(which mpirun)"
    echo "=== Environment ready ==="
    echo ""
}

# ---------- Pre-download CIFAR-10 ----------
download_data() {
    echo "=== Downloading CIFAR-10 data (if needed) ==="
    cd "$SCRIPT_DIR/examples"
    python -c "
import torchvision, os
d = os.path.join(os.getcwd(), 'datasets', 'RawData')
torchvision.datasets.CIFAR10(d, download=True, train=True)
torchvision.datasets.CIFAR10(d, download=True, train=False)
print('CIFAR-10 data ready at', d)
"
    cd "$SCRIPT_DIR"
    echo ""
}

# ---------- Run the experiment ----------
run_experiment() {
    echo "=== Running APPFL MPI DIMAT Paper Replication ==="
    echo "  Model         : ResNet20 (w=8, channels 128/256/512)"
    echo "  Server config : $SERVER_CONFIG"
    echo "  Client config : $CLIENT_CONFIG"
    echo "  MPI processes : $NUM_PROCS (1 server + $NUM_CLIENTS clients)"
    echo "  Partition     : non-IID, 2 classes per client"
    echo "  Optimizer     : Adam lr=0.001"
    echo "  Batch size    : 100"
    echo "  Local epochs  : 2 per round"
    echo "  Global rounds : 50"
    echo ""

    cd "$SCRIPT_DIR/examples"

    mpirun --oversubscribe -n "$NUM_PROCS" python mpi/run_mpi.py \
        --server_config "../$SERVER_CONFIG" \
        --client_config "../$CLIENT_CONFIG"

    cd "$SCRIPT_DIR"

    echo ""
    echo "=== Experiment finished ==="
    echo "Check examples/output/ for logs and results."
}

# ---------- Main ----------
main() {
    echo "============================================"
    echo " DIMAT Paper Replication: CIFAR-10 Non-IID"
    echo " 5 clients, fully connected, ResNet20"
    echo " $(date)"
    echo "============================================"
    echo ""

    verify_env
    download_data
    run_experiment
}

main
