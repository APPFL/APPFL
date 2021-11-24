#!/bin/bash
#
#SBATCH --job-name=mryu
#SBATCH --account=NEXTGENOPT
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=1:00:00
#SBATCH --array=1
#SBATCH --output=../Slurm_outputs/Slurm_outputs_run_%a.out
CASE_NUM=`printf %1d $SLURM_ARRAY_TASK_ID`

mpiexec -np 2 --mca opal_cuda_support 1 python ./mnist.py fed=fedavg num_epochs=1 batch_size=64 device=cuda
mpiexec -np 2 --mca opal_cuda_support 1 python ./mnist.py fed=iadmm num_epochs=1 batch_size=64 device=cuda
mpiexec -np 2 --mca opal_cuda_support 1 python ./femnist.py fed=fedavg num_epochs=1 batch_size=64 device=cuda
mpiexec -np 2 --mca opal_cuda_support 1 python ./femnist.py fed=iadmm num_epochs=1 batch_size=64 device=cuda
mpiexec -np 2 --mca opal_cuda_support 1 python ./coronahack.py fed=fedavg num_epochs=1 batch_size=64 device=cuda
mpiexec -np 2 --mca opal_cuda_support 1 python ./coronahack.py fed=iadmm num_epochs=1 batch_size=64 device=cuda
mpiexec -np 2 --mca opal_cuda_support 1 python ./cifar10.py fed=fedavg num_epochs=1 batch_size=64 device=cuda
mpiexec -np 2 --mca opal_cuda_support 1 python ./cifar10.py fed=iadmm num_epochs=1 batch_size=64 device=cuda