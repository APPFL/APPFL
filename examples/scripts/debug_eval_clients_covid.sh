#!/bin/bash
# python funcx_debug.py \
#     --client_config configs/clients/covid19_rad_database.yaml \
#     --config configs/fed_avg/funcx_fedavg_covid.yaml \
#     --clients-test \
#     --load-model \
#     --load-model-dirname log_funcx_appfl/server/outputs_CovidDataset_ServerFedAvg_Adam_funcx_fedavg_covid_covid19newsplit1_anl \
#     --load-model-filename checkpoint_30

# python funcx_debug.py \
#     --client_config configs/clients/covid19_rad_database.yaml \
#     --config configs/fed_avg/funcx_fedavg_covid.yaml \
#     --clients-test \
#     --load-model \
#     --load-model-dirname log_funcx_appfl/server/outputs_CovidDataset_ServerFedAvg_Adam_funcx_fedavg_covid_covid19_uchicago \
#     --load-model-filename checkpoint_30

# covid19newsplit2_anl

python funcx_debug.py \
    --client_config configs/clients/covid19_rad_database.yaml \
    --config configs/fed_avg/funcx_fedavg_covid_feat_extrc.yaml \
    --clients-test \
    --load-model \
    --load-model-dirname log_funcx_appfl/server/outputs_CovidDataset_ServerFedAvg_Adam_funcx_fedavg_covid_freeze_bn_covid19newsplit2_anl \
    --load-model-filename checkpoint_20