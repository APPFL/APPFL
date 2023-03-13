#!/bin/bash
# python funcx_debug.py \
#     --client_config configs/clients/covid19newsplit2_anl_uchicago_unnormalized.yaml \
#     --config configs/fed_avg/funcx_fedavg_covid.yaml \
#     --export-data-stats

python funcx_sync.py \
    --client_config configs/clients/covid19newsplit2_anl_uchicago_imgnet_norm.yaml \
    --config configs/fed_avg/funcx_fedavg_covid.yaml \
    --export-data-stats