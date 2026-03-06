#!/bin/bash
#SBATCH --job-name=appfl-dimat
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40GB
#SBATCH --time=02:00:00
#SBATCH --account=torch_pr_40_tandon_advanced
#SBATCH --output=appfl_dimat_%j.out
#SBATCH --error=appfl_dimat_%j.err

# ============================================================
# APPFL MPI DIMAT Test Script (CPU only, no GPU needed)
#
# Runs DIMAT with 1 server + 2 clients on CIFAR-10 using MPI.
# ============================================================

SCRIPT_DIR="/scratch/mp5847/src/APPFL"
cd "$SCRIPT_DIR"

# ---------- Activate conda environment ----------
eval "$(conda shell.bash hook)"
conda activate /scratch/mp5847/conda_environments/appfl

set -euo pipefail

# ---------- Configuration ----------
NUM_CLIENTS=2
NUM_PROCS=$((NUM_CLIENTS + 1))  # 1 server + N clients
SERVER_CONFIG="examples/resources/configs/cifar10/server_dimat.yaml"
CLIENT_CONFIG="examples/resources/configs/cifar10/client_dimat.yaml"

# ---------- Environment verification ----------
verify_env() {
    echo "=== Verifying environment ==="
    python -c "from mpi4py import MPI; print('mpi4py OK')"
    python -c "import appfl; print(f'appfl {appfl.__version__} OK')"
    python -c "from appfl.algorithm.aggregator import DIMATaggregator; print('DIMATaggregator OK')"
    python -c "from appfl.algorithm.trainer import DIMATTrainer; print('DIMATTrainer OK')"
    python -c "import scipy; print('scipy OK')"
    python -c "import networkx; print('networkx OK')"
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
    echo "=== Running APPFL MPI DIMAT on CIFAR-10 ==="
    echo "  Server config : $SERVER_CONFIG"
    echo "  Client config : $CLIENT_CONFIG"
    echo "  MPI processes : $NUM_PROCS (1 server + $NUM_CLIENTS clients)"
    echo ""

    # cd into examples/ since config paths are relative to it
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
    echo " APPFL MPI DIMAT on CIFAR-10 Test"
    echo " $(date)"
    echo "============================================"
    echo ""

    verify_env
    download_data
    run_experiment
}

main
