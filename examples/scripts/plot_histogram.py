from matplotlib import pyplot as plt
import numpy as np
import os.path as osp
import json
from scipy import stats as sts 
import seaborn as sns

log_dir = "log_funcx_appfl/server/"

viz_cfg = {
    "uiuc-cig-01-gpu-02": {
        "color": "blue",
        "name": "ANL Dataset"
    },
    "uchicago-gpu": {
        "color": "orange",
        "name": "UChicago Dataset"
    }

}

def do_plot(ax, dat, set, client, kde=True):
    hist, bins = dat[client][set]["hist"], dat[client][set]["bins"]
    if (kde == False):
        ax.bar(bins[:-1], hist, width=np.abs(bins[1]-bins[0]), color = viz_cfg[client]["color"])
    else:
        n=10000
        hist, bins = np.array(hist), np.array(bins)
        resamples = np.random.choice((bins[:-1] + bins[1:])/2, size=n, p=hist/hist.sum())
        sns.kdeplot(resamples, ax=ax, fill=False)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(0, 2.75)
    ax.grid("both")
    ax.set_title("%s - %s set" % (viz_cfg[client]["name"], set))
    ax.set_xlabel("Normalized Pixel Intensity")
    ax.set_ylabel("Probability")
        
def read_stats(fold, out_file):
    with open(osp.join(log_dir, fold, "data_stats.json")) as fi:
        dat = json.load(fi)
        f, axs = plt.subplots(1,2,figsize=(8,4))
        f.tight_layout(pad=5.0)
        for i, set in enumerate(["train", "val", "test"]):
            do_plot(axs[0], dat, set, client="uiuc-cig-01-gpu-02")
            do_plot(axs[1], dat, set, client="uchicago-gpu")
        plt.savefig(out_file, bbox_inches="tight")
    
read_stats(
    "outputs_CovidDataset_ServerFedAvg_Adam_funcx_fedavg_eqweight_covid_covid19newsplit2_anl_uchicago_data_norm",
    out_file="hist_normalized_data.png"
    )

read_stats(
    "outputs_CovidDataset_ServerFedAvg_Adam_funcx_fedavg_covid_covid19newsplit2_anl_uchicago_unnormalized",
    out_file="hist_unnormalized.png"
    )

read_stats(
    "outputs_CovidDataset_ServerFedAvg_Adam_funcx_fedavg_covid_covid19newsplit2_anl_uchicago_imgnet_norm",
    out_file="hist_normalized_imgnet.png"
    )
