#!/bin/bash
FL_FOLD="outputs_funcx_fedavg_covid_sgd_lr0.003_covid19_uchicago_v2_midrc_v2"
FL_CKPT="best"

python funcx_sync.py \
    --client_config configs/clients/covid19_midrc_v2.yaml \
    --config configs/tent/funcx_fedavg_covid_adapt_tent_adapt_test.yaml \
    --clients-adapt-test \
    --load-model \
    --load-model-dirname log_funcx_appfl/server/$FL_FOLD \
    --load-model-filename $FL_CKPT
