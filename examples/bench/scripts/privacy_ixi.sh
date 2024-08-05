#!/bin/bash
#SBATCH --mem=220g                              # required number of memory
#SBATCH --nodes=1                               # number of required nodes
#SBATCH --ntasks-per-node=1                     # number of tasks per node
#SBATCH --cpus-per-task=4                       # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA100x4                   # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bcwo-delta-gpu                # <- one of: bbke-delta-cpu or bbke-delta-gpu
#SBATCH --job-name=FLPrivacy                    # job name
#SBATCH --time=08:00:00                         # dd-hh:mm:ss for the job
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none                     

source ~/.bashrc
conda activate appfl
cd /projects/bcdz/zl52/APPFL/examples/bench

echo IXI-No-DP
python run_serial_privacy.py --server_config configs/privacy/ixi_server.yaml --client_config configs/privacy/ixi_client.yaml --num_clients 3
python run_serial_privacy.py --server_config configs/privacy/ixi_server.yaml --client_config configs/privacy/ixi_client.yaml --num_clients 3
python run_serial_privacy.py --server_config configs/privacy/ixi_server.yaml --client_config configs/privacy/ixi_client.yaml --num_clients 3
python run_serial_privacy.py --server_config configs/privacy/ixi_server.yaml --client_config configs/privacy/ixi_client.yaml --num_clients 3
python run_serial_privacy.py --server_config configs/privacy/ixi_server.yaml --client_config configs/privacy/ixi_client.yaml --num_clients 3
python run_serial_privacy.py --server_config configs/privacy/ixi_server.yaml --client_config configs/privacy/ixi_client.yaml --num_clients 3
python run_serial_privacy.py --server_config configs/privacy/ixi_server.yaml --client_config configs/privacy/ixi_client.yaml --num_clients 3
python run_serial_privacy.py --server_config configs/privacy/ixi_server.yaml --client_config configs/privacy/ixi_client.yaml --num_clients 3
python run_serial_privacy.py --server_config configs/privacy/ixi_server.yaml --client_config configs/privacy/ixi_client.yaml --num_clients 3
python run_serial_privacy.py --server_config configs/privacy/ixi_server.yaml --client_config configs/privacy/ixi_client.yaml --num_clients 3
echo EP-10
python run_serial_privacy.py --server_config configs/privacy/ixi_server_ep10.yaml --client_config configs/privacy/ixi_client.yaml --num_clients 3
python run_serial_privacy.py --server_config configs/privacy/ixi_server_ep10.yaml --client_config configs/privacy/ixi_client.yaml --num_clients 3
python run_serial_privacy.py --server_config configs/privacy/ixi_server_ep10.yaml --client_config configs/privacy/ixi_client.yaml --num_clients 3
python run_serial_privacy.py --server_config configs/privacy/ixi_server_ep10.yaml --client_config configs/privacy/ixi_client.yaml --num_clients 3
python run_serial_privacy.py --server_config configs/privacy/ixi_server_ep10.yaml --client_config configs/privacy/ixi_client.yaml --num_clients 3
python run_serial_privacy.py --server_config configs/privacy/ixi_server_ep10.yaml --client_config configs/privacy/ixi_client.yaml --num_clients 3
python run_serial_privacy.py --server_config configs/privacy/ixi_server_ep10.yaml --client_config configs/privacy/ixi_client.yaml --num_clients 3
python run_serial_privacy.py --server_config configs/privacy/ixi_server_ep10.yaml --client_config configs/privacy/ixi_client.yaml --num_clients 3
python run_serial_privacy.py --server_config configs/privacy/ixi_server_ep10.yaml --client_config configs/privacy/ixi_client.yaml --num_clients 3
python run_serial_privacy.py --server_config configs/privacy/ixi_server_ep10.yaml --client_config configs/privacy/ixi_client.yaml --num_clients 3
echo EP-1
python run_serial_privacy.py --server_config configs/privacy/ixi_server_ep1.yaml --client_config configs/privacy/ixi_client.yaml --num_clients 3
python run_serial_privacy.py --server_config configs/privacy/ixi_server_ep1.yaml --client_config configs/privacy/ixi_client.yaml --num_clients 3
python run_serial_privacy.py --server_config configs/privacy/ixi_server_ep1.yaml --client_config configs/privacy/ixi_client.yaml --num_clients 3
python run_serial_privacy.py --server_config configs/privacy/ixi_server_ep1.yaml --client_config configs/privacy/ixi_client.yaml --num_clients 3
python run_serial_privacy.py --server_config configs/privacy/ixi_server_ep1.yaml --client_config configs/privacy/ixi_client.yaml --num_clients 3
python run_serial_privacy.py --server_config configs/privacy/ixi_server_ep1.yaml --client_config configs/privacy/ixi_client.yaml --num_clients 3
python run_serial_privacy.py --server_config configs/privacy/ixi_server_ep1.yaml --client_config configs/privacy/ixi_client.yaml --num_clients 3
python run_serial_privacy.py --server_config configs/privacy/ixi_server_ep1.yaml --client_config configs/privacy/ixi_client.yaml --num_clients 3
python run_serial_privacy.py --server_config configs/privacy/ixi_server_ep1.yaml --client_config configs/privacy/ixi_client.yaml --num_clients 3
python run_serial_privacy.py --server_config configs/privacy/ixi_server_ep1.yaml --client_config configs/privacy/ixi_client.yaml --num_clients 3
echo EP-01
python run_serial_privacy.py --server_config configs/privacy/ixi_server_ep01.yaml --client_config configs/privacy/ixi_client.yaml --num_clients 3
python run_serial_privacy.py --server_config configs/privacy/ixi_server_ep01.yaml --client_config configs/privacy/ixi_client.yaml --num_clients 3
python run_serial_privacy.py --server_config configs/privacy/ixi_server_ep01.yaml --client_config configs/privacy/ixi_client.yaml --num_clients 3
python run_serial_privacy.py --server_config configs/privacy/ixi_server_ep01.yaml --client_config configs/privacy/ixi_client.yaml --num_clients 3
python run_serial_privacy.py --server_config configs/privacy/ixi_server_ep01.yaml --client_config configs/privacy/ixi_client.yaml --num_clients 3
python run_serial_privacy.py --server_config configs/privacy/ixi_server_ep01.yaml --client_config configs/privacy/ixi_client.yaml --num_clients 3
python run_serial_privacy.py --server_config configs/privacy/ixi_server_ep01.yaml --client_config configs/privacy/ixi_client.yaml --num_clients 3
python run_serial_privacy.py --server_config configs/privacy/ixi_server_ep01.yaml --client_config configs/privacy/ixi_client.yaml --num_clients 3
python run_serial_privacy.py --server_config configs/privacy/ixi_server_ep01.yaml --client_config configs/privacy/ixi_client.yaml --num_clients 3
python run_serial_privacy.py --server_config configs/privacy/ixi_server_ep01.yaml --client_config configs/privacy/ixi_client.yaml --num_clients 3

