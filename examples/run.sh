#!/bin/bash
#
#SBATCH --job-name=mryu
#SBATCH --account=NEXTGENOPT
#SBATCH --nodes=1
#SBATCH --gres=gpu:5
#SBATCH --time=10:00:00
#SBATCH --array=1
#SBATCH --output=../Slurm_outputs/Slurm_outputs_run_%a.out
CASE_NUM=`printf %1d $SLURM_ARRAY_TASK_ID`


mpiexec -np 5 --mca opal_cuda_support 1 python ./femnist.py fed=iadmm fed.args.num_local_epochs=10 fed.args.penalty=70.0 num_epochs=500 batch_size=64 device=cuda

