import sys
import os
import os.path as osp
import argparse
import subprocess

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("-c", "--checkpoint", type=str, default="best")
    parser.add_argument('-d', '--datasets', nargs='+', default=[])
    return parser.parse_args()

datasets = {
    "cxr2": {
        "cfg": "covid19newsplit2_anl.yaml"
    },
    "midrc_v2" :{
        "cfg": "covid19_midrc_v2.yaml"
    },
    "uchicago_v2": {
        "cfg": "covid19_uchicago_v2.yaml"
    },
    # "midrc": {
    #     "cfg": "covid19_midrc.yaml"
    # }
}
def main(args):
    
    print(args.output, args.checkpoint, args.datasets)
    for dts in args.datasets:
        print("=" * 50, "\n",
              "Experiment: ", args.output, "\n",
              "Checkpoint: ", args.checkpoint, "\n",
              "Dataset: ", dts, "\n",
              "=" * 50, "\n"
            )
        train_cfg = args.output.replace("outputs_","").split("_covid19")[0]
        
        cmds = [
            "python", "funcx_sync.py",
            "--clients-test",
            "--config",
            "configs/fed_avg/%s.yaml" % train_cfg,
            "--client_config", "configs/clients/%s" % datasets[dts]["cfg"],
            "--load-model",
            "--load-model-dirname", "log_funcx_appfl/server/%s" % args.output,
            "--load-model-filename", args.checkpoint
        ]
        subprocess.run(cmds)

if __name__ == "__main__":
    args = parse_args()    
    main(args)