#!/bin/bash
#SBATCH --mem=240g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu
#SBATCH --account=bbvf-delta-cpu                    
#SBATCH --job-name=grpc    
#SBATCH --time=02:00:00                        

source ~/.bashrc
conda activate appfl
cd /projects/bcdz/zl52/APPFL/examples/bench

python run_grpc_clients_serial_compression.py --num_clients 128 --server_uri 3.237.24.115:50051
sleep 60
python run_grpc_clients_serial_compression.py --num_clients 64 --server_uri 3.237.24.115:50051
sleep 60
python run_grpc_clients_serial_compression.py --num_clients 32 --server_uri 3.237.24.115:50051
sleep 60
python run_grpc_clients_serial_compression.py --num_clients 16 --server_uri 3.237.24.115:50051
sleep 60
python run_grpc_clients_serial_compression.py --num_clients 8 --server_uri 3.237.24.115:50051
sleep 60
python run_grpc_clients_serial_compression.py --num_clients 4 --server_uri 3.237.24.115:50051
sleep 60
python run_grpc_clients_serial_compression.py --num_clients 2 --server_uri 3.237.24.115:50051
