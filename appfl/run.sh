#!/bin/bash
#
#SBATCH --job-name=mryu
#SBATCH --account=NEXTGENOPT
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -w, --nodelist=gpu2
#SBATCH --time=10:00:00
#SBATCH --array=1
#SBATCH --output=./Slurm_outputs/Slurm_outputs_run_%a.out
CASE_NUM=`printf %1d $SLURM_ARRAY_TASK_ID`
srun python run_$CASE_NUM.py 