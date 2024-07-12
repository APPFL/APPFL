#!/bin/bash
#SBATCH --mem=240g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu
#SBATCH --account=bcwo-delta-cpu                    
#SBATCH --job-name=grpc-pse
#SBATCH --time=00:10:00                        

source ~/.bashrc
conda activate appfl
cd /projects/bcdz/zl52/APPFL/examples/bench
proxystore-endpoint start delta-proxystore-2
python run_grpc_clients_serial.py --num_clients 128 --client_config configs/communication/client_grpcpse.yaml --server_uri 141.142.144.66:50051
proxystore-endpoint stop delta-proxystore-2