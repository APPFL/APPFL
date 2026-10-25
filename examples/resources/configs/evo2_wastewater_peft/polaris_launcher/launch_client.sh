#!/bin/bash
ENV=/lus/grand/projects/GeomicVar/<username>/evo2_metagenomics/envs/evo2_env312
PY=/soft/applications/conda/2025-09-25/mconda3/bin/python3

export CUDA_HOME=/soft/compilers/cudatoolkit/cuda-12.6.0
export PATH="$CUDA_HOME/bin:$PATH"
CUDNN_LIBS="$ENV/lib/python3.12/site-packages/nvidia/cudnn/lib"
export CUDNN_PATH="$ENV/lib/python3.12/site-packages/nvidia/cudnn"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDNN_LIBS:${LD_LIBRARY_PATH:-}"
unset LD_PRELOAD PYTHONPATH

export PYTHONPATH="$ENV/lib/python3.12/site-packages"
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset http_proxy HTTP_PROXY
export https_proxy=http://140.221.69.93:3128 HTTPS_PROXY=http://140.221.69.93:3128
export no_proxy="localhost,127.0.0.1" NO_PROXY="localhost,127.0.0.1"

$PY -u /home/schunduru/APPFL-evo2-wastewater/examples/grpc/run_client_evo2.py \
    --config /home/schunduru/APPFL-evo2-wastewater/examples/resources/configs/evo2_wastewater_peft/client_polaris.yaml
