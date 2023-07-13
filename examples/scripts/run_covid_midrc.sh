#!/bin/bash
# python funcx_sync.py \
#     --client_config configs/clients/covid19_midrc.yaml \
#     --config configs/fed_avg/funcx_fedavg_covid_sgd_100epochs.yaml

python funcx_sync.py \
    --client_config configs/clients/covid19_midrc_v2.yaml \
    --config configs/fed_avg/funcx_fedavg_covid_sgd.yaml