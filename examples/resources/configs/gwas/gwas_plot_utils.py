import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CHROM_MAP = {"X": 23, "Y": 24, "XY": 25, "MT": 26, "M": 26}


def _normalize_chr(chrom):
    return chrom.astype(str).str.strip().str.upper().replace(CHROM_MAP).astype(int)


def _plot_manhattan(gwas_df, trait, threshold, out_path, label=""):
    n_snps = len(gwas_df)
    n_col = (
        "N_META"
        if "N_META" in gwas_df.columns
        else ("N" if "N" in gwas_df.columns else None)
    )
    n_samples = int(gwas_df[n_col].iloc[0]) if n_col else 0

    plot_df = gwas_df[["CHR", "BP", "P"]].copy()
    plot_df = plot_df.sort_values(["CHR", "BP"]).reset_index(drop=True)

    chrom_offsets = {}
    offset = 0
    tick_pos = []
    tick_labels = []
    for chrom, group in plot_df.groupby("CHR", sort=True):
        chrom = int(chrom)
        chrom_offsets[chrom] = offset
        bp = group["BP"].to_numpy(dtype=np.int64)
        tick_pos.append(offset + 0.5 * (bp.min() + bp.max()))
        tick_labels.append(str(chrom))
        offset += int(bp.max()) + 1_000_000

    plot_df["X"] = plot_df["BP"] + plot_df["CHR"].map(chrom_offsets)
    plot_df["LOGP"] = -np.log10(
        np.clip(plot_df["P"].to_numpy(dtype=np.float64), np.finfo(np.float64).tiny, 1.0)
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["#1f5a96", "#d76f30"]
    for idx, (_, group) in enumerate(plot_df.groupby("CHR", sort=True)):
        ax.scatter(
            group["X"],
            group["LOGP"],
            s=4,
            color=colors[idx % 2],
            alpha=0.8,
            linewidths=0,
        )
    ax.axhline(-np.log10(threshold), color="#9b1c31", linestyle="--", linewidth=1.2)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_xlabel("Chromosome")
    ax.set_ylabel("-log10(P)")
    ax.set_title(
        f"{trait} Manhattan Plot – {label}\nSNPs = {n_snps:,}  |  N = {n_samples:,}",
        fontsize=11,
    )
    ax.grid(axis="y", color="#dddddd", linewidth=0.8, alpha=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_qq(p_values, trait, max_points, out_path):
    p_values = np.asarray(p_values, dtype=np.float64)
    p_values = p_values[np.isfinite(p_values)]
    p_values = np.clip(p_values, np.finfo(np.float64).tiny, 1.0)
    if p_values.size > max_points:
        p_values = np.sort(p_values)[:max_points]
    else:
        p_values = np.sort(p_values)

    n = p_values.size
    expected = -np.log10((np.arange(1, n + 1) - 0.5) / n)
    observed = -np.log10(p_values)
    upper = max(float(expected.max()), float(observed.max()), 1.0)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(expected, observed, s=8, color="#1f5a96", alpha=0.75, linewidths=0)
    ax.plot([0, upper], [0, upper], color="#9b1c31", linestyle="--", linewidth=1.2)
    ax.set_xlim(0, upper * 1.03)
    ax.set_ylim(0, upper * 1.03)
    ax.set_xlabel("Expected -log10(P)")
    ax.set_ylabel("Observed -log10(P)")
    ax.set_title(f"{trait} QQ Plot")
    ax.grid(color="#dddddd", linewidth=0.8, alpha=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _write_hits_table(bmi_df, t2d_df, hit_threshold, out_path):
    hit_tables = []
    for trait_df in [bmi_df, t2d_df]:
        hits = trait_df.loc[trait_df["P"] < hit_threshold].copy()
        if hits.empty:
            hits = trait_df.nsmallest(100, "P").copy()
            hits["HIT_SET"] = "top100"
        else:
            hits = hits.sort_values("P").copy()
            hits["HIT_SET"] = f"p<{hit_threshold:g}"
        hit_tables.append(hits)
    pd.concat(hit_tables, ignore_index=True).to_csv(out_path, index=False)
