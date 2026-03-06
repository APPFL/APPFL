#!/bin/bash
#SBATCH --job-name=dimat-orig
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80GB
#SBATCH --time=08:00:00
#SBATCH --account=torch_pr_40_tandon_advanced
#SBATCH --output=dimat_orig_%j.out
#SBATCH --error=dimat_orig_%j.err

# ============================================================
# Original DIMAT baseline: CIFAR-100, IID, 5 agents, Ring
# Paper reference (Table 2): DIMAT Ring = 67.12 ± 0.22%
# ============================================================

SCRIPT_DIR="/scratch/mp5847/src/APPFL"
cd "$SCRIPT_DIR"

eval "$(conda shell.bash hook)"
conda activate /scratch/mp5847/conda_environments/appfl

set -euo pipefail

# Install missing deps if needed
pip install scipy scikit-learn 2>/dev/null || true

echo "============================================"
echo " Original DIMAT Baseline"
echo " CIFAR-100, IID, 5 agents, Ring"
echo " Paper target: 67.12 ± 0.22%"
echo " $(date)"
echo "============================================"

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

python run_original_dimat.py \
    --pretrain_epochs 100 \
    --merge_rounds 100 \
    --train_epochs 2 \
    --batch_size 100 \
    --num_models 5 \
    --seed 42

echo "=== Done ==="
