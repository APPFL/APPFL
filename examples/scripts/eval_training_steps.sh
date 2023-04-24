#!/bin/bash
TRAIN_CFG="configs/fed_avg/funcx_fedavg_covid_sgd.yaml"
MIDRC_FOLD="outputs_funcx_fedavg_covid_sgd_covid19_midrc"

for i in {2..30..2}
do
  echo "checkpoint_$i"
  python funcx_sync.py \
    --client_config configs/clients/covid19_midrc_train_debug.yaml \
    --config $TRAIN_CFG \
    --clients-test \
    --load-model \
    --load-model-dirname log_funcx_appfl/server/$MIDRC_FOLD \
    --load-model-filename "checkpoint_$i"
done