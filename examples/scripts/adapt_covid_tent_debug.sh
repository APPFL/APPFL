#!/bin/bash
FL_FOLD="outputs_funcx_fedavg_covid_sgd_lr0.003_covid19_uchicago_v2_midrc_v2"
FL_CKPT="best"

python funcx_debug.py \
    --client_config configs/clients/covid19newsplit2_anl.yaml \
    --config configs/tent/funcx_fedavg_covid_adapt_tent_debug.yaml \
    --clients-test \
    --load-model \
    --load-model-dirname log_funcx_appfl/server/$FL_FOLD \
    --load-model-filename $FL_CKPT