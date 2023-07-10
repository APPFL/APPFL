from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf
import os

@dataclass
class FDA:
    type: str = "federatedda"
    servername: str = "ServerFDA"
    clientname: str = "ClientOptim"
    # target: int = 0
    # n_target_samples: int = 2000
    # source_batch_size: int = 16
    # target_batch_size: int = 16
    args: DictConfig = OmegaConf.create(
        {
            ## Server update
            "server_learning_rate": 0.01,
            "server_adapt_param": 0.001,
            "server_momentum_param_1": 0.9,
            "server_momentum_param_2": 0.99,
            ## Clients optimizer
            "optim": "Adam",
            "num_local_epochs": 10,
            "optim_args": {
                "lr": 0.001,
            },
            ## FDA
            "target_lr_ratio": 0.2,
            "n_target_samples": 2000,
            "source_batch_size": 16,
            "target_batch_size": 8,
            "target": 0,
            ## Differential Privacy
            ##  epsilon: False  (non-private)
            ##  epsilon: 1      (stronger privacy as the value decreases)
            ##  epsilon: 0.05
            "epsilon": False,
            ## Gradient Clipping
            ## clip_value: False (no-clipping)
            ## clip_value: 10    (clipping)
            ## clip_value: 1
            "clip_value": False,
            "clip_norm": 1,
        }
    )
    
    

# ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# SHARED_DIR = '/shared/rsaas/enyij2/'

# DATA_DIR = os.path.join(SHARED_DIR, 'midrc', 'data')
# META_DATA_DIR = 'meta_data_info'
# STATE_DATA_DIR = os.path.join(META_DATA_DIR, 'states')
# CHEXPERT_DATA_DIR = os.path.join(META_DATA_DIR, 'chexpert')
# MIMIC_DATA_DIR = os.path.join(META_DATA_DIR, 'mimic_small')
# # DEMO_DATA_DIR = os.path.join(SHARED_DIR, 'CXR', 'data_demo')
# MSDA_BASE_DIR = os.path.join(SHARED_DIR, 'msda')

# SOURCE_TRAIN_CSV_PATH = os.path.join(META_DATA_DIR, 'MIDRC_cr_table_all_train.csv')
# TARGET_TRAIN_CSV_PATH = os.path.join(META_DATA_DIR, 'MIDRC_dx_table_all_train.csv')
# SOURCE_TEST_CSV_PATH = os.path.join(META_DATA_DIR, 'MIDRC_cr_table_all_test.csv')
# TARGET_TEST_CSV_PATH = os.path.join(META_DATA_DIR, 'MIDRC_dx_table_all_test.csv')

# IL_TRAIN_CSV_PATH = os.path.join(STATE_DATA_DIR, 'MIDRC_table_IL_train.csv')
# IL_TEST_CSV_PATH = os.path.join(STATE_DATA_DIR, 'MIDRC_table_IL_test.csv')
# NC_TRAIN_CSV_PATH = os.path.join(STATE_DATA_DIR, 'MIDRC_table_NC_train.csv')
# NC_TEST_CSV_PATH = os.path.join(STATE_DATA_DIR, 'MIDRC_table_NC_test.csv')
# CA_TRAIN_CSV_PATH = os.path.join(STATE_DATA_DIR, 'MIDRC_table_CA_train.csv')
# CA_TEST_CSV_PATH = os.path.join(STATE_DATA_DIR, 'MIDRC_table_CA_test.csv')
# IN_TRAIN_CSV_PATH = os.path.join(STATE_DATA_DIR, 'MIDRC_table_IN_train.csv')
# IN_TEST_CSV_PATH = os.path.join(STATE_DATA_DIR, 'MIDRC_table_IN_test.csv')
# TX_TRAIN_CSV_PATH = os.path.join(STATE_DATA_DIR, 'MIDRC_table_TX_train.csv')
# TX_TEST_CSV_PATH = os.path.join(STATE_DATA_DIR, 'MIDRC_table_TX_test.csv')