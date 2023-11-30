import argparse
from appfl.config import *
from appfl.misc.data import *
from globus_compute_sdk import Client
from appfl.run_globus_compute_server import run_server
from appfl.comm.globus_compute.utils.logging import GlobusComputeServerLogger

"""
python superglue_globus_compute.py --client_config path_to_client_config.yaml --server_config path_to_server_config.yaml
"""

""" read arguments """ 
parser = argparse.ArgumentParser()  
# parser.add_argument("--client_config", type=str, default="globus_compute/configs_client/superglue_cb.yaml")
# parser.add_argument("--server_config", type=str, default= "globus_compute/configs_server/superglue_cb_fedavg.yaml")
parser.add_argument("--client_config", type=str, default="globus_compute/configs_client/superglue_copa.yaml")
parser.add_argument("--server_config", type=str, default= "globus_compute/configs_server/superglue_copa_fedavg.yaml")

args = parser.parse_args()

def main():
    # Configuration    
    cfg = OmegaConf.structured(GlobusComputeConfig)
    cfg.validation = True   

    # loading globus compute configs from file
    load_globus_compute_client_config(cfg, args.client_config)
    load_globus_compute_server_config(cfg, args.server_config)
    GlobusComputeServerLogger.config_logger(cfg)    
    
    server_test_dataset = None
    server_val_dataset  = None
    gcc = Client()
    run_server(cfg, None, None, None, gcc, server_test_dataset, server_val_dataset)

if __name__ == "__main__":
    main()