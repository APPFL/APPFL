python funcx_sync.py \
    --client_config configs/clients/covid19newsplit3_anl.yaml \
    --config configs/fed_avg/funcx_fedavg_covid.yaml \
    --clients-test \
    --load-model \
    --load-model-dirname log_funcx_appfl/server/outputs_funcx_fedavg_covid_sgd_covid19_midrc_v2 \
    --load-model-filename best