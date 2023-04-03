#!/bin/bash
# python funcx_sync.py \
#     --client_config configs/clients/covid19_uchicago.yaml \
#     --config configs/fed_avg/funcx_fedavg_covid.yaml

# python funcx_sync.py \
    # --client_config configs/clients/covid19_uchicago.yaml \
    # --config configs/fed_avg/funcx_fedavg_covid_resnet50.yaml

# python funcx_sync.py \
#     --client_config configs/clients/covid19_uchicago.yaml \
#     --config configs/fed_avg/funcx_fedavg_covid_freeze_bn.yaml

python funcx_sync.py \
    --client_config configs/clients/covid19_uchicago.yaml \
    --config configs/fed_avg/funcx_fedavg_covid_sgd.yaml