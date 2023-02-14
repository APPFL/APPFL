#!/bin/bash
# python funcx_sync.py \
#     --client_config configs/clients/covid19_anl_uchicago.yaml \
#     --config configs/fed_avg/funcx_fedavg_eqweight_covid.yaml \
#     --clients-test \
#     --load-model \
#     --load-model-dirname log_funcx_appfl/server/outputs_CovidDataset_ServerFedAvg_Adam_funcx_fedavg_eqweight_covid_covid19_anl_uchicago \
#     --load-model-filename checkpoint_30

python funcx_sync.py \
    --client_config configs/clients/covid19newsplit1_anl_uchicago.yaml \
    --config configs/fed_avg/funcx_fedavg_covid.yaml \
    --clients-test \
    --load-model \
    --load-model-dirname log_funcx_appfl/server/outputs_CovidDataset_ServerFedAvg_Adam_funcx_fedavg_covid_covid19newsplit1_anl \
    --load-model-filename checkpoint_30