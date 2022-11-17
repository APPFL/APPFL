#!/bin/bash
python funcx_sync.py \
    --client_config configs/clients/ecg_anl_broad.yaml \
    --config configs/fed_avg/funcx_fedavg_ecg_resnet.yaml \
    --clients-test \
    --load-model \
    --load-model-dirname log_funcx_appfl/server/outputs_ECGDataset_ServerFedAvg_Adam_funcx_fedavg_ecg_resnet_ecg_broad \
    --load-model-filename checkpoint_30