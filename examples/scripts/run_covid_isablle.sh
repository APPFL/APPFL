#!/bin/bash

# python funcx_debug.py \
#     --client_config configs/clients/covid19newsplit2_anl.yaml \
#     --config configs/fed_avg/funcx_fedavg_covid_adam_isabelle.yaml \
#     --load-model \
#     --load-model-dirname pretrained \
#     --load-model-filename isabelle_sigmoid_state_dict

# python funcx_sync.py \
#     --client_config configs/clients/covid19_midrc.yaml \
#     --config configs/fed_avg/funcx_fedavg_covid_adam_isabelle.yaml \
#     --load-model \
#     --load-model-dirname pretrained \
#     --load-model-filename isabelle_sigmoid_state_dict

python funcx_sync.py \
    --client_config configs/clients/covid19_uchicago_v2.yaml \
    --config configs/fed_avg/funcx_fedavg_covid_adam_isabelle.yaml \
    --load-model \
    --load-model-dirname pretrained \
    --load-model-filename isabelle_sigmoid_state_dict