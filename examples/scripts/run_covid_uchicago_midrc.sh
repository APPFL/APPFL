#!/bin/bash
# python funcx_sync.py \
#     --client_config configs/clients/covid19_uchicago_v2_midrc.yaml \
#     --config configs/fed_avg/funcx_fedavg_covid_sgd_lr0.003.yaml

# python funcx_sync.py \
#     --client_config configs/clients/covid19_uchicago_v2_midrc.yaml \
#     --config configs/fed_avg/funcx_fedavg_covid_sgd_lr0.005_gn.yaml

python funcx_sync.py \
    --client_config configs/clients/covid19_uchicago_v2_midrc_v2.yaml \
    --config configs/fed_avg/funcx_fedavg_covid_sgd_lr0.003.yaml