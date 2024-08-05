#!/bin/bash
#SBATCH --mem=240g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu
#SBATCH --account=bbvf-delta-cpu                    
#SBATCH --job-name=grpc    
#SBATCH --time=01:30:00                        

source ~/.bashrc
conda activate appfl
cd /projects/bcdz/zl52/APPFL/examples/bench
python run_grpc_clients_serial.py --num_clients 128 --server_uri 3.237.24.115:50051