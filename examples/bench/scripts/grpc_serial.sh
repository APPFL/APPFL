#!/bin/bash
#SBATCH --mem=240g                                  # required number of memory
#SBATCH --nodes=1                                   # number of required nodes
#SBATCH --ntasks-per-node=1                         # number of tasks per node
#SBATCH --cpus-per-task=4                           # <- match to OMP_NUM_THREADS
#SBATCH --partition=cpu                             # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bcwo-delta-cpu                    
#SBATCH --job-name=grpc-128-clients    
#SBATCH --time=00:30:00                        

source ~/.bashrc
conda activate appfl
cd /projects/bcdz/zl52/APPFL/examples/bench
python run_grpc_clients_serial.py --num_clients 96 --server_uri 141.142.144.67:50051