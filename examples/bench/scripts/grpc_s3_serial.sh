#!/bin/bash
#SBATCH --mem=240g 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu
#SBATCH --account=bcwo-delta-cpu                    
#SBATCH --job-name=grpc-128-clients    
#SBATCH --time=00:10:00                        

source ~/.bashrc
conda activate appfl
cd /projects/bcdz/zl52/APPFL/examples/bench
python run_grpc_clients_serial.py --num_clients 8 --client_config configs/communication/client_grpcs3.yaml --server_uri 141.142.144.162:50051
