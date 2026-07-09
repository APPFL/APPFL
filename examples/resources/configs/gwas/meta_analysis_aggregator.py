# ruff: noqa: E402
#   Thread-count env vars and sys.path must be set before importing numpy /
#   gwas_config, so imports are intentionally not all at the top of the file.
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

_gwas_demo_dir = os.environ.get("GWAS_PROJECT_DIR")
if _gwas_demo_dir:
    sys.path.insert(0, _gwas_demo_dir)
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from gwas_config import HIT_P_THRESHOLD

import json
import numpy as np
import pandas as pd
import torch
from appfl.algorithm.aggregator import BaseAggregator
from gwas_plot_utils import _plot_manhattan, _plot_qq, _write_hits_table
from scipy.stats import norm


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class MetaAnalysisAggregator(BaseAggregator):
    def __init__(self, model=None, aggregator_configs=None, logger=None):
        self.model = model
        self.logger = logger
        self.aggregator_configs = aggregator_configs
        self.hit_threshold = float(
            aggregator_configs.get("hit_p_threshold", HIT_P_THRESHOLD)
        )
        self.qq_max_points = int(aggregator_configs.get("qq_max_points", 250000))
        self.output_dir = Path(
            aggregator_configs.get("output_dir", "local/server_output")
        ).resolve()
        self.data_dir = self.output_dir / "data"
        self.graphs_dir = self.output_dir / "graphs"
        self.logs_dir = self.output_dir / "logs"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.graphs_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.global_state = {
            "meta_ready": torch.tensor([0], dtype=torch.int64),
        }

    def get_parameters(self, **kwargs):
        return self.global_state

    def aggregate(self, local_models, **kwargs):
        client_ids = list(local_models.keys())
        self.logger.info(
            f"Running fixed-effect meta-analysis across {len(client_ids)} APPFL sites."
        )

        _meta_bytes = _to_numpy(local_models[client_ids[0]]["variant_meta"]).tobytes()
        variant_df = pd.DataFrame(json.loads(_meta_bytes.decode("utf-8")))

        bmi_beta_stack = np.vstack(
            [_to_numpy(local_models[cid]["bmi_beta"]) for cid in client_ids]
        )
        bmi_se_stack = np.vstack(
            [_to_numpy(local_models[cid]["bmi_se"]) for cid in client_ids]
        )
        t2d_beta_stack = np.vstack(
            [_to_numpy(local_models[cid]["t2d_beta"]) for cid in client_ids]
        )
        t2d_se_stack = np.vstack(
            [_to_numpy(local_models[cid]["t2d_se"]) for cid in client_ids]
        )
        maf_stack = np.vstack(
            [_to_numpy(local_models[cid]["maf"]) for cid in client_ids]
        )
        gwas_n = np.array(
            [int(_to_numpy(local_models[cid]["gwas_n"])[0]) for cid in client_ids],
            dtype=np.float64,
        )
        eval_n = np.array(
            [int(_to_numpy(local_models[cid]["eval_n"])[0]) for cid in client_ids],
            dtype=np.float64,
        )
        bmi_r2 = np.array(
            [
                float(_to_numpy(local_models[cid]["local_bmi_r2"])[0])
                for cid in client_ids
            ],
            dtype=np.float64,
        )
        t2d_auc = np.array(
            [
                float(_to_numpy(local_models[cid]["local_t2d_auc"])[0])
                for cid in client_ids
            ],
            dtype=np.float64,
        )

        n_variants = len(variant_df)
        if bmi_beta_stack.shape[1] != n_variants:
            raise ValueError(
                f"Client payload variant count ({bmi_beta_stack.shape[1]}) does not match "
                f"variant metadata from {client_ids[0]} ({n_variants})."
            )

        total_n = int(gwas_n.sum())
        meta_maf = np.average(maf_stack, axis=0, weights=gwas_n)
        bmi_df = self._meta_analyze_trait(
            bmi_beta_stack, bmi_se_stack, meta_maf, "BMI", total_n, variant_df
        )
        t2d_df = self._meta_analyze_trait(
            t2d_beta_stack, t2d_se_stack, meta_maf, "T2D", total_n, variant_df
        )

        bmi_path = self.data_dir / "appfl_meta_gwas_bmi.csv.gz"
        t2d_path = self.data_dir / "appfl_meta_gwas_t2d.csv.gz"
        hits_path = self.data_dir / "appfl_meta_gwas_hits.csv"
        metrics_path = self.data_dir / "appfl_site_pgs_metrics.csv"
        summary_path = self.data_dir / "appfl_meta_summary.csv"

        bmi_df.to_csv(bmi_path, index=False)
        t2d_df.to_csv(t2d_path, index=False)
        _write_hits_table(bmi_df, t2d_df, self.hit_threshold, hits_path)
        self._write_site_metrics(
            client_ids, gwas_n, eval_n, bmi_r2, t2d_auc, metrics_path, summary_path
        )

        _plot_manhattan(
            bmi_df,
            "BMI",
            self.hit_threshold,
            self.graphs_dir / "appfl_meta_gwas_bmi_manhattan.png",
            label="APPFL",
        )
        _plot_qq(
            bmi_df["P"].to_numpy(dtype=np.float64),
            "BMI",
            self.qq_max_points,
            self.graphs_dir / "appfl_meta_gwas_bmi_qq.png",
        )
        _plot_manhattan(
            t2d_df,
            "T2D",
            self.hit_threshold,
            self.graphs_dir / "appfl_meta_gwas_t2d_manhattan.png",
            label="APPFL",
        )
        _plot_qq(
            t2d_df["P"].to_numpy(dtype=np.float64),
            "T2D",
            self.qq_max_points,
            self.graphs_dir / "appfl_meta_gwas_t2d_qq.png",
        )

        self.global_state = {
            "meta_ready": torch.tensor([1], dtype=torch.int64),
            "num_clients": torch.tensor([len(client_ids)], dtype=torch.int64),
            "num_variants": torch.tensor([n_variants], dtype=torch.int64),
            "bmi_hits": torch.tensor(
                [int((bmi_df["P"] < self.hit_threshold).sum())], dtype=torch.int64
            ),
            "t2d_hits": torch.tensor(
                [int((t2d_df["P"] < self.hit_threshold).sum())], dtype=torch.int64
            ),
            "weighted_local_bmi_r2": torch.tensor(
                [float(np.average(bmi_r2, weights=eval_n))], dtype=torch.float64
            ),
            "weighted_local_t2d_auc": torch.tensor(
                [float(np.average(t2d_auc, weights=eval_n))], dtype=torch.float64
            ),
        }
        self.logger.info(
            f"APPFL meta-analysis outputs written to {self.output_dir}/{{data,graphs}}"
        )
        return self.global_state

    def _meta_analyze_trait(
        self, beta_stack, se_stack, maf, trait, total_n, variant_df
    ):
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = np.where(
                np.isfinite(se_stack) & (se_stack > 0), 1.0 / np.square(se_stack), 0.0
            )
        weight_sum = weights.sum(axis=0)
        beta = np.divide(
            np.sum(weights * beta_stack, axis=0),
            weight_sum,
            out=np.full(weight_sum.shape, np.nan, dtype=np.float64),
            where=weight_sum > 0,
        )
        se = np.sqrt(
            np.divide(
                1.0,
                weight_sum,
                out=np.full(weight_sum.shape, np.nan, dtype=np.float64),
                where=weight_sum > 0,
            )
        )
        stat = np.divide(
            beta,
            se,
            out=np.zeros(weight_sum.shape, dtype=np.float64),
            where=np.isfinite(se) & (se > 0),
        )
        p_value = np.clip(
            2.0 * norm.sf(np.abs(stat)),
            np.finfo(np.float64).tiny,
            1.0,
        )

        out_df = variant_df.copy()
        out_df["TRAIT"] = trait
        out_df["BETA"] = beta
        out_df["SE"] = se
        out_df["STAT"] = stat
        if trait == "T2D":
            out_df["OR"] = np.exp(np.clip(beta, -50, 50))
        out_df["P"] = p_value
        out_df["MAF"] = maf
        out_df["N_META"] = total_n
        return out_df

    def _write_site_metrics(
        self, client_ids, gwas_n, eval_n, bmi_r2, t2d_auc, metrics_path, summary_path
    ):
        site_metrics = pd.DataFrame(
            {
                "CLIENT_ID": client_ids,
                "GWAS_N": gwas_n.astype(int),
                "EVAL_N": eval_n.astype(int),
                "LOCAL_BMI_R2": bmi_r2,
                "LOCAL_T2D_AUROC": t2d_auc,
            }
        )
        site_metrics.to_csv(metrics_path, index=False)

        summary = pd.DataFrame(
            [
                {
                    "NUM_CLIENTS": len(client_ids),
                    "TOTAL_GWAS_N": int(gwas_n.sum()),
                    "TOTAL_EVAL_N": int(eval_n.sum()),
                    "WEIGHTED_LOCAL_BMI_R2": float(np.average(bmi_r2, weights=eval_n)),
                    "WEIGHTED_LOCAL_T2D_AUROC": float(
                        np.average(t2d_auc, weights=eval_n)
                    ),
                }
            ]
        )
        summary.to_csv(summary_path, index=False)
