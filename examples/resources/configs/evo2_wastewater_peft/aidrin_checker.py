"""
AIDRIN data-readiness gate for the FL pipeline.

Called once at trainer init — before the training dataset is built — to
identify contaminated samples.  Contaminated files are added to the
trainer's exclude_files list so they never enter the gradient loop.

If every training file is contaminated the trainer skips its local update
and sends a sentinel parameter dict to the server.  The server aggregator
detects the sentinel and excludes that client from FedAvg.

Contamination criterion (primary):
    drisee_score_raw > drisee_threshold  →  contaminated
    drisee_score_raw is NaN              →  contaminated (conservative)

AIDRIN Python API is attempted first; falls back to pandas/numpy metrics
(identical to run_aidrin_local.py) when the API is unavailable.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Defaults — override per-site via train_configs YAML
DRISEE_THRESHOLD = 20.0

# Sentinel key sent to server when a site has no clean training data
AIDRIN_SKIP_KEY = "__aidrin_skip__"


def _mgm_id_from_fasta(filename: str) -> str:
    """mgm4467420.3_mgm4467420.3.299.screen.passed.fna → mgm4467420.3"""
    return Path(filename).name.split("_")[0]


def _aidrin_meta_metrics(csv_path: Path) -> Dict:
    """Run AIDRIN (or fallback) on the site CSV and return a summary dict."""
    try:
        from aidrin import Inspector
        inspector = Inspector(data=str(csv_path), label=None)
        result = inspector.assess(
            metrics=["completeness", "duplicates", "outliers", "correlations"]
        )
        d = result.to_dict() if hasattr(result, "to_dict") else {}
        d["_source"] = "aidrin_python_api"
        return d
    except Exception:
        df = pd.read_csv(csv_path)
        n = len(df)
        completeness = float(1 - df.isnull().values.mean())
        dup_frac = int(df.duplicated().sum()) / n if n else 0.0
        num = df.select_dtypes(include=[np.number])
        outlier_fracs = []
        for col in num.columns:
            q1, q3 = num[col].quantile(0.25), num[col].quantile(0.75)
            iqr = q3 - q1
            n_out = int(
                ((num[col] < q1 - 1.5 * iqr) | (num[col] > q3 + 1.5 * iqr)).sum()
            )
            outlier_fracs.append(n_out / n if n else 0.0)
        return {
            "completeness": round(completeness, 4),
            "duplicate_fraction": round(dup_frac, 4),
            "outlier_fraction": round(float(np.mean(outlier_fracs)) if outlier_fracs else 0.0, 4),
            "_source": "fallback_pandas",
        }


def check_site_data(
    site: str,
    aidrin_csv: Path,
    fasta_dir: Path,
    current_exclude: List[str],
    max_files: Optional[int] = None,
    drisee_threshold: float = DRISEE_THRESHOLD,
    report_dir: Optional[Path] = None,
    logger=None,
) -> Tuple[List[str], List[str], Dict]:
    """
    Assess training data quality for one FL site.

    Args:
        site:             site name (for logging)
        aidrin_csv:       path to <site>_aidrin_inputs.csv
        fasta_dir:        directory containing .fna/.fastq training files
        current_exclude:  already-excluded file patterns (held-out set)
        max_files:        cap on training files (same as dataset config)
        drisee_threshold: max acceptable DRISEE score
        report_dir:       if set, write JSON report here
        logger:           APPFL logger (falls back to print)

    Returns:
        (new_exclude_patterns, contaminated_fnames, report_dict)
        new_exclude_patterns: mgm_ids to ADD to exclude_files (contaminated)
        contaminated_fnames:  full filenames that were contaminated
        report_dict:          full AIDRIN assessment report
    """

    def log(msg: str):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    # --- Resolve candidate training files (mirrors FastaChunkDataset logic) ---
    all_fasta = sorted(
        list(fasta_dir.glob("*.fna")) + list(fasta_dir.glob("*.fastq"))
    )
    candidates = [
        f for f in all_fasta
        if not any(ex in f.name for ex in current_exclude)
    ]
    if max_files:
        candidates = sorted(candidates, key=lambda f: f.stat().st_size)[:max_files]

    if not candidates:
        log("[AIDRIN] No candidate training files to assess.")
        return [], [], {"site": site, "skipped": True, "reason": "no_candidates"}

    # --- If CSV not found, pass all files through ---
    if not aidrin_csv.exists():
        log(f"[AIDRIN] CSV not found ({aidrin_csv}) — all files treated as clean")
        return [], [], {"site": site, "skipped": True, "reason": "csv_not_found"}

    # --- AIDRIN meta-assessment on the full CSV ---
    log(f"[AIDRIN] Running assessment on {aidrin_csv.name} ...")
    meta = _aidrin_meta_metrics(aidrin_csv)
    log(
        f"[AIDRIN] CSV quality — completeness={meta.get('completeness','?')}, "
        f"dup_frac={meta.get('duplicate_fraction','?')}, "
        f"outlier_frac={meta.get('outlier_fraction','?')} "
        f"(source={meta.get('_source','?')})"
    )

    # --- Build mgm_id → DRISEE lookup ---
    df = pd.read_csv(aidrin_csv)
    if "mgm_id" in df.columns and "drisee_score_raw" in df.columns:
        drisee_lookup = df.set_index("mgm_id")["drisee_score_raw"].to_dict()
    else:
        log("[AIDRIN] CSV missing mgm_id or drisee_score_raw — treating all as clean")
        return [], [], {"site": site, "skipped": True, "reason": "missing_columns"}

    # --- Per-file contamination check ---
    new_excludes: List[str] = []
    contaminated_fnames: List[str] = []
    clean_fnames: List[str] = []
    per_file: Dict = {}

    for fpath in candidates:
        fname = fpath.name
        mgm_id = _mgm_id_from_fasta(fname)
        drisee = drisee_lookup.get(mgm_id, None)

        if drisee is None:
            status, reason = "contaminated", f"mgm_id '{mgm_id}' not in AIDRIN CSV"
        elif isinstance(drisee, float) and np.isnan(drisee):
            status, reason = "contaminated", f"drisee_score_raw is NaN"
        elif float(drisee) > drisee_threshold:
            status = "contaminated"
            reason = f"drisee_score_raw={drisee:.2f} > threshold={drisee_threshold}"
        else:
            status, reason = "clean", f"drisee_score_raw={drisee:.2f}"

        per_file[fname] = {"mgm_id": mgm_id, "status": status, "reason": reason}
        log(f"[AIDRIN] {mgm_id}: {status} — {reason}")

        if status == "contaminated":
            new_excludes.append(mgm_id)
            contaminated_fnames.append(fname)
        else:
            clean_fnames.append(fname)

    site_ready = len(clean_fnames) > 0
    log(
        f"[AIDRIN] {site}: {len(clean_fnames)} clean, "
        f"{len(contaminated_fnames)} contaminated — "
        f"{'READY' if site_ready else 'SKIPPING (all contaminated)'}"
    )

    report = {
        "site": site,
        "aidrin_csv": str(aidrin_csv),
        "drisee_threshold": drisee_threshold,
        "csv_meta_metrics": meta,
        "n_candidates": len(candidates),
        "n_clean": len(clean_fnames),
        "n_contaminated": len(contaminated_fnames),
        "clean_files": clean_fnames,
        "contaminated_files": contaminated_fnames,
        "per_file": per_file,
        "site_ready": site_ready,
    }

    if report_dir is not None:
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"{site}_fl_aidrin_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        log(f"[AIDRIN] Report → {report_path}")

    return new_excludes, contaminated_fnames, report
