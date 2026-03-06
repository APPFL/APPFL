#!/bin/bash
#SBATCH --job-name=appfl-mnist
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80GB
#SBATCH --time=23:00:00
#SBATCH --account=torch_pr_40_tandon_advanced
#SBATCH --output=appfl_mnist_%j.out
#SBATCH --error=appfl_mnist_%j.err

# ============================================================
# APPFL MPI MNIST Test Script (SLURM)
#
# Runs FedAvg with 1 server + 3 clients on MNIST using MPI.
#
# Usage:
#   1. Interactive (if MPI is available):
#        bash test_mpi_mnist.sh --local
#
#   2. Submit to SLURM:
#        sbatch test_mpi_mnist.sh
#
# Adjust the #SBATCH directives above for your cluster
# (partition, account, time limit, etc.) as needed.
# ============================================================

SCRIPT_DIR="/scratch/mp5847/src/APPFL"
cd "$SCRIPT_DIR"

# ---------- Activate conda environment ----------
# Activate before set -u; conda scripts reference unbound variables
eval "$(conda shell.bash hook)"
conda activate /scratch/mp5847/conda_environments/appfl

set -euo pipefail

# ---------- Configuration ----------
NUM_CLIENTS=3
NUM_PROCS=$((NUM_CLIENTS + 1))  # 1 server + N clients
SERVER_CONFIG="examples/resources/configs/mnist/server_fedavg.yaml"
CLIENT_CONFIG="examples/resources/configs/mnist/client_1.yaml"

# ---------- Environment verification ----------
verify_env() {
    echo "=== Verifying environment ==="
    python -c "from mpi4py import MPI; print('mpi4py OK')"
    python -c "import appfl; print(f'appfl {appfl.__version__} OK')"
    python -c "import torchvision; print('torchvision OK')"
    echo "mpirun: $(which mpirun)"
    echo "=== Environment ready ==="
    echo ""
}

# ---------- Pre-download MNIST ----------
download_data() {
    echo "=== Downloading MNIST data (if needed) ==="
    python -c "
import torchvision, os
d = os.path.join(os.getcwd(), 'datasets', 'RawData')
torchvision.datasets.MNIST(d, download=True, train=True)
torchvision.datasets.MNIST(d, download=True, train=False)
print('MNIST data ready at', d)
"
    echo ""
}

# ---------- Run the experiment ----------
run_experiment() {
    echo "=== Running APPFL MPI FedAvg on MNIST ==="
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
    echo "Check output/ for logs and results."
}

# ---------- Main ----------
main() {
    echo "============================================"
    echo " APPFL MPI MNIST Test"
    echo " $(date)"
    echo "============================================"
    echo ""

    verify_env
    download_data
    run_experiment
}

# If running interactively with --local, just call main directly.
# Under SLURM (sbatch), main is called at the bottom regardless.
main
