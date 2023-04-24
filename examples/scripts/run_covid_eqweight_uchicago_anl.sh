#!/bin/bash
# python funcx_sync.py \
#     --client_config configs/clients/covid19newsplit2_anl_uchicago_data_norm.yaml \
#     --config configs/fed_avg/funcx_fedavg_eqweight_covid.yaml \
#     --export-data-stats

python funcx_sync.py \
    --client_config configs/clients/covid19newsplit2_anl_uchicago_eqprevalence.yaml \
    --config configs/fed_avg/funcx_eqweight_covid_sgd_freeze_bn.yaml \