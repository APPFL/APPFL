#!/bin/bash
TRAIN_CFG="configs/fed_avg/funcx_fedavg_covid_sgd.yaml"

# ANL_FOLD="outputs_CovidDataset_ServerFedAvg_SGD_funcx_fedavg_covid_sgd_covid19newsplit2_anl"
# ANL_CKPT="best"

ANL_FOLD="outputs_funcx_fedavg_covid_sgd_lr0.005_covid19newsplit2_anl"
ANL_CKPT="checkpoint_40"

# UC_FOLD="outputs_funcx_fedavg_covid_sgd_covid19_uchicago_eqprevalence"
# UC_CKPT="best"
# UC_FOLD="outputs_funcx_fedavg_covid_sgd_lr0.005_covid19_uchicago_eqprevalence"
UC_FOLD="outputs_funcx_fedavg_covid_sgd_covid19_uchicago_v2"
UC_CKPT="best"

# ANL_UC_FOLD="outputs_funcx_fedavg_covid_sgd_freeze_bn_covid19newsplit2_anl_uchicago_eqprevalence"
# ANL_UC_FOLD="outputs_funcx_eqweight_covid_sgd_freeze_bn_covid19newsplit2_anl_uchicago_eqprevalence"
ANL_UC_FOLD="outputs_CovidDataset_ServerFedAvg_SGD_funcx_fedavg_covid_sgd_covid19newsplit2_anl_uchicago_imgnet_norm"
ANL_UC_CKPT="best"
# ANL_UC_FOLD="outputs_funcx_fedavg_covid_sgd_lr0.005_covid19newsplit2_anl_uchicago_eqprevalence"
# ANL_UC_CKPT="checkpoint_24"

# MIDRC_FOLD="outputs_funcx_fedavg_covid_sgd_covid19_midrc"
# MIDRC_CKPT="checkpoint_30"

MIDRC_FOLD="outputs_funcx_fedavg_covid_sgd_100epochs_covid19_midrc"
MIDRC_CKPT="checkpoint_34"

ANL_MIDRC_FOLD="outputs_funcx_fedavg_covid_sgd_covid19newsplit2_anl_midrc_eqprevalence"
ANL_MIDRC_CKPT="best"

UC_MIDRC_FOLD="outputs_funcx_fedavg_covid_sgd_lr0.003_covid19_uchicago_eqprevalence_midrc"
UC_MIDRC_CKPT="best"

# ================ CXR 2 Test set (Test set 1) ================
# python funcx_sync.py \
#     --client_config configs/clients/covid19newsplit2_anl.yaml \
#     --config $TRAIN_CFG \
#     --clients-test \
#     --load-model \
#     --load-model-dirname log_funcx_appfl/server/$ANL_UC_FOLD \
#     --load-model-filename $ANL_UC_CKPT

# python funcx_sync.py \
#     --client_config configs/clients/covid19newsplit2_anl.yaml \
#     --config $TRAIN_CFG \
#     --clients-test \
#     --load-model \
#     --load-model-dirname log_funcx_appfl/server/$ANL_FOLD \
#     --load-model-filename $ANL_CKPT

# python funcx_sync.py \
#     --client_config configs/clients/covid19newsplit2_anl.yaml \
#     --config $TRAIN_CFG \
#     --clients-test \
#     --load-model \
#     --load-model-dirname log_funcx_appfl/server/$UC_FOLD \
#     --load-model-filename $UC_CKPT

# python funcx_sync.py \
#     --client_config configs/clients/covid19newsplit2_anl.yaml \
#     --config $TRAIN_CFG \
#     --clients-test \
#     --load-model \
#     --load-model-dirname log_funcx_appfl/server/$MIDRC_FOLD \
#     --load-model-filename $MIDRC_CKPT

# python funcx_sync.py \
#     --client_config configs/clients/covid19newsplit2_anl.yaml \
#     --config $TRAIN_CFG \
#     --clients-test \
#     --load-model \
#     --load-model-dirname log_funcx_appfl/server/$ANL_MIDRC_FOLD \
#     --load-model-filename $ANL_MIDRC_CKPT

# ================ COVID-19 Rad Database set  (Test set 2) ================
# python funcx_debug.py \
#     --client_config configs/clients/covid19_rad_database.yaml \
#     --config $TRAIN_CFG \
#     --clients-test \
#     --load-model \
#     --load-model-dirname log_funcx_appfl/server/$ANL_UC_FOLD \
#     --load-model-filename $ANL_UC_CKPT

# python funcx_debug.py \
#     --client_config configs/clients/covid19_rad_database.yaml \
#     --config $TRAIN_CFG \
#     --clients-test \
#     --load-model \
#     --load-model-dirname log_funcx_appfl/server/$ANL_FOLD \
#     --load-model-filename $ANL_CKPT

# python funcx_debug.py \
#     --client_config configs/clients/covid19_rad_database.yaml \
#     --config $TRAIN_CFG \
#     --clients-test \
#     --load-model \
#     --load-model-dirname log_funcx_appfl/server/$UC_FOLD \
#     --load-model-filename $UC_CKPT

# python funcx_sync.py \
#     --client_config configs/clients/covid19newsplit2_anl.yaml \
#     --config $TRAIN_CFG \
#     --clients-test \
#     --load-model \
#     --load-model-dirname log_funcx_appfl/server/$UC_MIDRC_FOLD \
#     --load-model-filename $UC_MIDRC_CKPT

# ================ COVID19-TEH Dataset (Test set 3) ================

# python funcx_debug.py \
#     --client_config configs/clients/covid19_teh.yaml \
#     --config $TRAIN_CFG \
#     --clients-test \
#     --load-model \
#     --load-model-dirname log_funcx_appfl/server/$ANL_UC_FOLD \
#     --load-model-filename $ANL_UC_CKPT

# python funcx_debug.py \
#     --client_config configs/clients/covid19_teh.yaml \
#     --config $TRAIN_CFG \
#     --clients-test \
#     --load-model \
#     --load-model-dirname log_funcx_appfl/server/$ANL_FOLD \
#     --load-model-filename $ANL_CKPT

# python funcx_debug.py \
#     --client_config configs/clients/covid19_teh.yaml \
#     --config $TRAIN_CFG \
#     --clients-test \
#     --load-model \
#     --load-model-dirname log_funcx_appfl/server/$UC_FOLD \
#     --load-model-filename $UC_CKPT

# ================ UChicago Test set ================
# python funcx_sync.py \
#     --client_config configs/clients/covid19_uchicago_eqprevalence.yaml \
#     --config $TRAIN_CFG \
#     --clients-test \
#     --load-model \
#     --load-model-dirname log_funcx_appfl/server/$ANL_UC_FOLD \
#     --load-model-filename $ANL_UC_CKPT

# python funcx_sync.py \
#     --client_config configs/clients/covid19_uchicago_eqprevalence.yaml \
#     --config $TRAIN_CFG \
#     --clients-test \
#     --load-model \
#     --load-model-dirname log_funcx_appfl/server/$ANL_FOLD \
#     --load-model-filename $ANL_CKPT

# python funcx_sync.py \
#     --client_config configs/clients/covid19_uchicago_eqprevalence.yaml \
#     --config $TRAIN_CFG \
#     --clients-test \
#     --load-model \
#     --load-model-dirname log_funcx_appfl/server/$UC_FOLD \
#     --load-model-filename $UC_CKPT


# python funcx_sync.py \
#     --client_config configs/clients/covid19_uchicago_eqprevalence.yaml \
#     --config configs/fed_avg/funcx_eqweight_covid_freeze_bn.yaml \
#     --clients-test \
#     --load-model \
#     --load-model-dirname log_funcx_appfl/server/outputs_CovidDataset_ServerFedAvg_Adam_funcx_fedavg_covid_covid19_uchicago \
#     --load-model-filename best

# python funcx_sync.py \
#     --client_config configs/clients/covid19_uchicago_eqprevalence.yaml \
#     --config $TRAIN_CFG \
#     --clients-test \
#     --load-model \
#     --load-model-dirname log_funcx_appfl/server/$ANL_MIDRC_FOLD \
#     --load-model-filename $ANL_MIDRC_CKPT

# python funcx_sync.py \
#     --client_config configs/clients/covid19_uchicago_eqprevalence.yaml \
#     --config $TRAIN_CFG \
#     --clients-test \
#     --load-model \
#     --load-model-dirname log_funcx_appfl/server/$ANL_UC_FOLD \
#     --load-model-filename $ANL_UC_CKPT

# python funcx_sync.py \
#     --client_config configs/clients/covid19_uchicago_eqprevalence.yaml \
#     --config $TRAIN_CFG \
#     --clients-test \
#     --load-model \
#     --load-model-dirname log_funcx_appfl/server/$MIDRC_FOLD \
#     --load-model-filename $MIDRC_CKPT

# ================ UChicago v2 ================
python funcx_sync.py \
    --client_config configs/clients/covid19_uchicago_v2.yaml \
    --config $TRAIN_CFG \
    --clients-test \
    --load-model \
    --load-model-dirname log_funcx_appfl/server/$UC_FOLD \
    --load-model-filename $UC_CKPT

# ================ MIDRCTest set ================
# python funcx_sync.py \
#     --client_config configs/clients/covid19_midrc.yaml \
#     --config $TRAIN_CFG \
#     --clients-test \
#     --load-model \
#     --load-model-dirname log_funcx_appfl/server/$ANL_UC_FOLD \
#     --load-model-filename $ANL_UC_CKPT

# python funcx_sync.py \
#     --client_config configs/clients/covid19_midrc.yaml \
#     --config $TRAIN_CFG \
#     --clients-test \
#     --load-model \
#     --load-model-dirname log_funcx_appfl/server/$ANL_FOLD \
#     --load-model-filename $ANL_CKPT

# python funcx_sync.py \
#     --client_config configs/clients/covid19_midrc.yaml \
#     --config $TRAIN_CFG \
#     --clients-test \
#     --load-model \
#     --load-model-dirname log_funcx_appfl/server/$UC_FOLD \
#     --load-model-filename $UC_CKPT

# python funcx_sync.py \
#     --client_config configs/clients/covid19_midrc.yaml \
#     --config $TRAIN_CFG \
#     --clients-test \
#     --load-model \
#     --load-model-dirname log_funcx_appfl/server/$MIDRC_FOLD \
#     --load-model-filename $MIDRC_CKPT

# python funcx_sync.py \
#     --client_config configs/clients/covid19_midrc.yaml \
#     --config $TRAIN_CFG \
#     --clients-test \
#     --load-model \
#     --load-model-dirname log_funcx_appfl/server/$ANL_MIDRC_FOLD \
#     --load-model-filename $ANL_MIDRC_CKPT


# ================ UChicago Train Debug ================

# python funcx_sync.py \
#     --client_config configs/clients/covid19_uchicago_eqprevalence_train_debug.yaml \
#     --config $TRAIN_CFG \
#     --clients-test \
#     --load-model \
#     --load-model-dirname log_funcx_appfl/server/$UC_FOLD \
#     --load-model-filename $UC_CKPT

# python funcx_sync.py \
#     --client_config configs/clients/covid19_uchicago_eqprevalence_train_debug.yaml \
#     --config $TRAIN_CFG \
#     --clients-test \
#     --load-model \
#     --load-model-dirname log_funcx_appfl/server/$ANL_UC_FOLD \
#     --load-model-filename $ANL_UC_CKPT

# python funcx_sync.py \
#     --client_config configs/clients/covid19_uchicago_eqprevalence_train_debug.yaml \
#     --config $TRAIN_CFG \
#     --clients-test \
#     --load-model \
#     --load-model-dirname log_funcx_appfl/server/$UC_MIDRC_FOLD \
#     --load-model-filename $UC_MIDRC_CKPT

# python funcx_sync.py \
#     --client_config configs/clients/covid19_midrc_train_debug.yaml \
#     --config $TRAIN_CFG \
#     --clients-test \
#     --load-model \
#     --load-model-dirname log_funcx_appfl/server/$MIDRC_FOLD \
#     --load-model-filename $MIDRC_CKPT

# python funcx_sync.py \
#     --client_config configs/clients/covid19_midrc_train_debug.yaml \
#     --config $TRAIN_CFG \
#     --clients-test \
#     --load-model \
#     --load-model-dirname log_funcx_appfl/server/$ANL_MIDRC_FOLD \
#     --load-model-filename $ANL_MIDRC_CKPT

# python funcx_sync.py \
#     --client_config configs/clients/covid19_midrc_train_debug.yaml \
#     --config $TRAIN_CFG \
#     --clients-test \
#     --load-model \
#     --load-model-dirname log_funcx_appfl/server/$UC_MIDRC_FOLD \
#     --load-model-filename $UC_MIDRC_CKPT

# python funcx_sync.py \
#     --client_config configs/clients/covid19newsplit2_anl_train_debug.yaml \
#     --config $TRAIN_CFG \
#     --clients-test \
#     --load-model \
#     --load-model-dirname log_funcx_appfl/server/$ANL_UC_FOLD \
#     --load-model-filename $ANL_UC_CKPT

# python funcx_sync.py \
#     --client_config configs/clients/covid19newsplit2_anl_train_debug.yaml \
#     --config $TRAIN_CFG \
#     --clients-test \
#     --load-model \
#     --load-model-dirname log_funcx_appfl/server/$ANL_MIDRC_FOLD \
#     --load-model-filename $ANL_MIDRC_CKPT