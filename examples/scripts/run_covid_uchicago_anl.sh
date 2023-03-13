#!/bin/bash
# python funcx_sync.py \
#     --client_config configs/clients/covid19_anl_uchicago.yaml \
#     --config configs/fed_avg/funcx_fedavg_covid.yaml

# python funcx_sync.py \
#     --client_config configs/clients/covid19newsplit2_anl_uchicago_imgnet_norm.yaml \
#     --config configs/fed_avg/funcx_fedavg_covid_feat_extrc.yaml

# python funcx_sync.py \
#     --client_config configs/clients/covid19newsplit2_anl_uchicago_data_norm.yaml \
#     --config configs/fed_avg/funcx_eqweight_covid_resnet.yaml

# python funcx_sync.py \
#     --client_config configs/clients/covid19newsplit2_anl_data_norm.yaml \
#     --config configs/fed_avg/funcx_fedavg_covid.yaml

# python funcx_sync.py \
#     --client_config configs/clients/covid19_uchicago_data_norm.yaml \
#     --config configs/fed_avg/funcx_fedavg_covid.yaml

python funcx_sync.py \
    --client_config configs/clients/covid19newsplit2_anl_uchicago_imgnet_norm.yaml \
    --config configs/fed_avg/funcx_fedavg_covid_freeze_bn.yaml