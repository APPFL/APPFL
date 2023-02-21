#!/bin/bash
python funcx_debug.py \
    --client_config configs/clients/covid19newsplit2_anl.yaml \
    --config configs/model_attack/funcx_fedavg_covid_no_laplace_1epoch.yaml \
    --clients-privacy-attack \
    # --load-model \
    # --load-model-dirname log_funcx_appfl/server/outputs_CovidDataset_ServerFedAvg_Adam_funcx_fedavg_covid_covid19_anl_uchicago \
    # --load-model-filename checkpoint_30