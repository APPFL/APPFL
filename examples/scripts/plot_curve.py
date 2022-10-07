from cProfile import label
import json
import argparse
import os.path as osp
from matplotlib import pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--logfile', type=str, required=True, help='path to log file')
    args   = parser.parse_args()
    return args

def main(args):
    with open(args.logfile) as fi:
        data = json.load(fi)
    
    val_clients_dict = data["val"]["clients"]
    val_clients_arr = {}
    for i, v in enumerate(val_clients_dict):
        for client_name in v:
            if client_name not in val_clients_arr:
                val_clients_arr[client_name] = {}
            for item in v[client_name]:
                if item not in val_clients_arr[client_name]:
                    val_clients_arr[client_name][item] = []
                val_clients_arr[client_name][item].append(
                    val_clients_dict[i][client_name][item]
                )
    
    f, axs = plt.subplots(1, 2, figsize=(18,7))
    f.suptitle(osp.basename(osp.dirname(args.logfile)), fontsize=15)
    skip = 1
    for client_name in val_clients_arr:
        axs[0].plot(
            val_clients_arr[client_name]['step'][skip:],
            val_clients_arr[client_name]['val_acc'][skip:],
            label = client_name
        )
    axs[0].set_xlabel("Global step")
    axs[0].set_ylabel("Val Accuracy")
    axs[0].legend(loc="upper right")
    axs[0].set_xticks(val_clients_arr[client_name]['step'][skip:])
    axs[0].set_xticklabels(val_clients_arr[client_name]['step'][skip:])
    for client_name in val_clients_arr:
        axs[1].plot(
            val_clients_arr[client_name]['step'][skip:],
            val_clients_arr[client_name]['val_loss'][skip:],
            label = client_name
        )
    axs[1].set_xlabel("Global step")
    axs[1].set_ylabel("Val Loss")
    axs[1].legend(loc="upper right")
    axs[1].set_xticks(val_clients_arr[client_name]['step'][skip:])
    axs[1].set_xticklabels(val_clients_arr[client_name]['step'][skip:])
    axs[0].grid()
    axs[1].grid()
    
    plt.savefig(osp.join(osp.dirname(args.logfile), "val_plot.png"), bbox_inches="tight")
     
if __name__ == "__main__":
    args = parse_args()
    main(args)