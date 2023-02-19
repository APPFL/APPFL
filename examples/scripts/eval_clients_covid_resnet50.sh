#!/bin/bash

# python funcx_sync.py \
#     --client_config configs/clients/covid19newsplit2_anl_uchicago.yaml \
#     --config configs/fed_avg/funcx_fedavg_covid_resnet50.yaml \
#     --clients-test \
#     --load-model \
#     --load-model-dirname log_funcx_appfl/server/outputs_CovidDataset_ServerFedAvg_Adam_funcx_fedavg_covid_resnet50_covid19newsplit2_anl \
#     --load-model-filename checkpoint_30

python funcx_sync.py \
    --client_config configs/clients/covid19newsplit2_anl_uchicago.yaml \
    --config configs/fed_avg/funcx_fedavg_covid_resnet50.yaml \
    --clients-test \
    --load-model \
    --load-model-dirname log_funcx_appfl/server/outputs_CovidDataset_ServerFedAvg_Adam_funcx_fedavg_covid_resnet50_covid19_uchicago \
    --load-model-filename checkpoint_30