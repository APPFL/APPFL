#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
python funcx_debug.py \
    --client_config configs/clients/covid19newsplit2_anl.yaml \
    --config configs/fed_avg/funcx_fedavg_covid_sgd.yaml