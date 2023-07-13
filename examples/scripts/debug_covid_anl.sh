#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
# python funcx_sync.py \
#     --client_config configs/clients/covid19newsplit2_anl.yaml \
#     --config configs/fed_avg/funcx_fedavg_covid_sgd_100epochs.yaml

# python funcx_sync.py \
#     --client_config configs/clients/covid19newsplit2_anl.yaml \
#     --config configs/fed_avg/funcx_fedavg_covid_sgd_100epochs.yaml

python funcx_debug.py \
    --client_config configs/clients/covid19newsplit2_anl.yaml \
    --config configs/fed_avg/funcx_fedavg_covid_sgd_gn.yaml