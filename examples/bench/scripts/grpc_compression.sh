#!/bin/bash
#SBATCH --mem=240g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu
#SBATCH --account=bcwo-delta-cpu                    
#SBATCH --job-name=grpc    
#SBATCH --time=00:30:00                        

source ~/.bashrc
conda activate appfl
cd /projects/bcdz/zl52/APPFL/examples/bench
python run_grpc_clients_serial_compression.py --num_clients 64 --server_uri 141.142.144.99:50051