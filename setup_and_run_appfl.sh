#!/bin/bash

# Update and install required packages
sudo apt update -y
sudo apt install python3-pip -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt install python3.10 python3.10-venv python3.10-distutils -y

# Verify Python version
python3.10 --version

# Create and activate a virtual environment
python3.10 -m venv my_env
source my_env/bin/activate

# Clone the repository and install dependencies
git clone --single-branch --branch zey/webinar https://github.com/APPFL/APPFL.git
cd APPFL
pip install -e ".[dev,examples]"

# Navigate to examples directory and run the server and clients
cd examples

# Run server

python ./grpc/run_server.py --config ./resources/configs/mnist/server_fedcompass.yaml

# Run clients

python ./grpc/run_client.py --config ./resources/configs/mnist/client_1.yaml

python ./grpc/run_client.py --config ./resources/configs/mnist/client_2.yaml
