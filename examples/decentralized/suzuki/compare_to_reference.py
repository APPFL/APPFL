"""Compare APPFL Suzuki results against the ADKO reference outputs.

    python examples/decentralized/suzuki/compare_to_reference.py \
        --appfl     results/appfl_main \
        --reference ../adko/scientific_discovery/results/main/<combo> \
        --arm IID

The comparison is paired by seed: both runs should use the same warmup bank, so
differences are attributable to the implementation rather than initialization.
Final yield is not enough because Suzuki often saturates at 100%; the useful
signals are rounds-to-threshold, the ``system_best`` curve, and token fidelity
``eta_bar``.

Terminology:

* one seed is one full run;
* one run has 200 rounds, and each round has one evaluation per agent;
* ``system_best[t]`` is one number for one seed: the best yield found by any
  agent up to round ``t``;
* the mean ``system_best`` curve averages ``system_best[t]`` across seeds at
  each fixed round ``t``;
* the AUC-style summary first averages ``system_best`` over rounds within each
  seed, then compares those paired seed-level averages.

Use ``--strict-steps`` when checking implementation equivalence. It reports the
first mismatch in sensitive per-step fields instead of only aggregate metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

THRESHOLDS = (90.0, 95.0, 99.0, 100.0)
DEFAULT_N_BOOT = 5000
DEFAULT_CI = 95.0


def load_runs(directory: Path, arm: str) -> Dict[int, dict]:
    """``{seed: result}`` for one arm (``IID`` or ``HET``) of ADKO runs in a directory."""
    runs: Dict[int, dict] = {}
    for path in sorted(directory.glob(f"{arm}-ADKO_seed*.json")):
        payload = json.loads(path.read_text())
        runs[int(payload["seed"])] = payload
    if not runs:
        raise SystemExit(f"no {arm}-ADKO_seed*.json found in {directory}")
    return runs


def jsonable(value: Any) -> Any:
    """Convert numpy values into JSON-safe Python values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def rounds_to(curve: np.ndarray, threshold: float) -> Optional[int]:
    """First round at which ``system_best`` reaches ``threshold``; ``None`` if never."""
    hit = np.flatnonzero(curve >= threshold)
    return int(hit[0]) if hit.size else None


def format_hits(values: List[Optional[int]]) -> str:
    got = [v for v in values if v is not None]
    if not got:
        return "never"
    tail = f" ({len(values) - len(got)} never)" if len(got) < len(values) else ""
    return f"median {np.median(got):5.1f}  mean {np.mean(got):6.2f}{tail}"


def ci_percentiles(level: float) -> Tuple[float, float]:
    """Convert a CI level into lower/upper percentile cutoffs."""
    tail = (100.0 - level) / 2.0
    return tail, 100.0 - tail


def bootstrap_mean_ci(
    values: Sequence[float],
    n_boot: int = DEFAULT_N_BOOT,
    level: float = DEFAULT_CI,
    seed: int = 0,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for the mean of one vector."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.array([x[rng.integers(0, x.size, x.size)].mean() for _ in range(n_boot)])
    lo, hi = ci_percentiles(level)
    return float(np.percentile(means, lo)), float(np.percentile(means, hi))


def paired_diff_ci(
    appfl_values: Sequence[float],
    reference_values: Sequence[float],
    n_boot: int = DEFAULT_N_BOOT,
    level: float = DEFAULT_CI,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Mean paired difference ``APPFL - reference`` with bootstrap CI."""
    diff = np.asarray(appfl_values, dtype=float) - np.asarray(reference_values, dtype=float)
    lo, hi = bootstrap_mean_ci(diff, n_boot=n_boot, level=level, seed=seed)
    return float(diff.mean()), lo, hi


def warmup_y_matrix(payload: dict) -> np.ndarray:
    """``(warmup_rounds, n_agents)`` matrix of per-agent warmup yields."""
    cfg = payload["config"]
    n_rounds = int(cfg["warmup_rounds"])
    n_agents = int(cfg["n_agents"])
    out = np.full((n_rounds, n_agents), np.nan)
    for step in payload["steps"]:
        round_idx = int(step["round"])
        if round_idx >= n_rounds:
            continue
        out[round_idx, int(step["agent"])] = float(step["y_raw"])
    return out


def system_best_matrix(runs: Dict[int, dict], seeds: Sequence[int]) -> np.ndarray:
    """Rows are seeds, columns are rounds."""
    return np.array([runs[seed]["system_best"] for seed in seeds], dtype=float)


def bootstrap_ci(
    samples: np.ndarray,
    n_boot: int = DEFAULT_N_BOOT,
    level: float = DEFAULT_CI,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Percentile bootstrap CI of the mean curve, over the seed axis."""
    rng = np.random.default_rng(seed)
    n = samples.shape[0]
    means = np.stack(
        [samples[rng.integers(0, n, n)].mean(axis=0) for _ in range(n_boot)]
    )
    lo, hi = ci_percentiles(level)
    return np.percentile(means, lo, axis=0), np.percentile(means, hi, axis=0)


def step_field_matrix(payload: dict, field: str, default: float = np.nan) -> np.ndarray:
    """``(rounds, agents)`` matrix for a numeric per-step field."""
    cfg = payload["config"]
    n_rounds = int(cfg["n_rounds"])
    n_agents = int(cfg["n_agents"])
    out = np.full((n_rounds, n_agents), default, dtype=float)
    for step in payload["steps"]:
        value = step.get(field, default)
        if value is None:
            continue
        out[int(step["round"]), int(step["agent"])] = float(value)
    return out


def step_values(payload: dict, field: str) -> Dict[Tuple[int, int], object]:
    """Per-step field keyed by ``(round, agent)``."""
    return {
        (int(step["round"]), int(step["agent"])): step.get(field)
        for step in payload["steps"]
    }


def first_step_mismatch(
    appfl_payload: dict,
    reference_payload: dict,
    fields: Iterable[str],
    atol: float,
) -> Optional[Tuple[str, int, int, object, object]]:
    """Return first per-step mismatch for selected fields, if one exists."""
    for field in fields:
        a_values = step_values(appfl_payload, field)
        r_values = step_values(reference_payload, field)
        for key in sorted(set(a_values) & set(r_values)):
            a, r = a_values[key], r_values[key]
            if isinstance(a, (float, int)) or isinstance(r, (float, int)):
                mismatch = abs(float(a) - float(r)) > atol
            else:
                mismatch = a != r
            if mismatch:
                return field, key[0], key[1], a, r
    return None


def hit_pairs(
    appfl_curves: np.ndarray, reference_curves: np.ndarray, threshold: float
) -> Tuple[List[float], List[float], int, int]:
    """Rounds-to-threshold values where both sides hit, plus one-sided misses."""
    appfl_hits = [rounds_to(curve, threshold) for curve in appfl_curves]
    reference_hits = [rounds_to(curve, threshold) for curve in reference_curves]
    appfl_only_miss = sum(a is None and r is not None for a, r in zip(appfl_hits, reference_hits))
    ref_only_miss = sum(a is not None and r is None for a, r in zip(appfl_hits, reference_hits))
    pairs = [(float(a), float(r)) for a, r in zip(appfl_hits, reference_hits)
             if a is not None and r is not None]
    if not pairs:
        return [], [], appfl_only_miss, ref_only_miss
    appfl_values, reference_values = zip(*pairs)
    return list(appfl_values), list(reference_values), appfl_only_miss, ref_only_miss


def print_config_warnings(appfl_cfg: dict, ref_cfg: dict) -> List[dict]:
    """Warn when high-level experiment settings differ."""
    keys = (
        "beta",
        "lam",
        "gamma",
        "tau",
        "token_budget",
        "alpha_tau",
        "p_noise",
        "similarity_bandwidth",
        "n_rounds",
        "warmup_rounds",
    )
    mismatches = []
    for key in keys:
        appfl_value, ref_value = appfl_cfg.get(key), ref_cfg.get(key)
        if appfl_value != ref_value:
            mismatches.append({
                "key": key,
                "appfl": appfl_value,
                "reference": ref_value,
            })
            print(
                f"  !! config mismatch on {key}: "
                f"appfl={appfl_value} reference={ref_value}"
            )
    return mismatches


def print_warmup_check(
    appfl: Dict[int, dict],
    reference: Dict[int, dict],
    seeds: Sequence[int],
) -> dict:
    """Confirm the paired runs start from the same warmup observations."""
    warm_gaps = [
        float(np.nanmax(np.abs(warmup_y_matrix(appfl[s]) - warmup_y_matrix(reference[s]))))
        for s in seeds
    ]
    warm_gap = max(warm_gaps)
    identical = warm_gap < 1e-9
    status = "shared warmup confirmed" if identical else "NOT identical"
    print(f"warmup per-agent y_raw max |diff| across seeds: {warm_gap:.3g}   <- {status}")
    return {
        "max_abs_diff": warm_gap,
        "identical": identical,
        "per_seed_max_abs_diff": dict(zip(seeds, warm_gaps)),
    }


def print_threshold_table(
    appfl_curves: np.ndarray,
    reference_curves: np.ndarray,
    n_boot: int,
    ci_level: float,
) -> Dict[str, dict]:
    """Report paired rounds-to-threshold differences."""
    report = {}
    ci_label = f"{ci_level:.0f}%"
    print("\nrounds to reach yield threshold")
    print(f"  {'thresh':>7} | {'APPFL':>34} | {'reference':>34} | paired APPFL-ref")
    for threshold in THRESHOLDS:
        appfl_hits = [rounds_to(curve, threshold) for curve in appfl_curves]
        ref_hits = [rounds_to(curve, threshold) for curve in reference_curves]
        paired_appfl, paired_ref, appfl_miss, ref_miss = hit_pairs(
            appfl_curves, reference_curves, threshold
        )
        if paired_appfl:
            mean_diff, lo, hi = paired_diff_ci(
                paired_appfl,
                paired_ref,
                n_boot=n_boot,
                level=ci_level,
                seed=int(threshold),
            )
            same = int(np.sum(np.asarray(paired_appfl) == np.asarray(paired_ref)))
            diff_report = {
                "mean": mean_diff,
                "ci": [lo, hi],
                "identical": same,
                "paired_hits": len(paired_appfl),
            }
            diff = (
                f"mean {mean_diff:+.2f}  {ci_label} CI [{lo:+.2f}, {hi:+.2f}]  "
                f"identical {same}/{len(paired_appfl)}"
            )
        else:
            diff_report = None
            diff = "n/a"
        if appfl_miss or ref_miss:
            diff += f"  misses appfl={appfl_miss} ref={ref_miss}"
        report[str(int(threshold))] = {
            "appfl": {
                "rounds": appfl_hits,
                "median": None if not paired_appfl else float(np.median(paired_appfl)),
                "mean": None if not paired_appfl else float(np.mean(paired_appfl)),
            },
            "reference": {
                "rounds": ref_hits,
                "median": None if not paired_ref else float(np.median(paired_ref)),
                "mean": None if not paired_ref else float(np.mean(paired_ref)),
            },
            "paired_diff": diff_report,
            "misses": {
                "appfl_only": appfl_miss,
                "reference_only": ref_miss,
            },
        }
        print(
            f"  {threshold:7.0f} | {format_hits(appfl_hits):>34} | "
            f"{format_hits(ref_hits):>34} | {diff}"
        )
    return report


def curve_summary(
    appfl_curves: np.ndarray,
    reference_curves: np.ndarray,
    n_boot: int,
    ci_level: float,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    dict,
]:
    """Print curve-level agreement and return arrays used for plotting."""
    ci_label = f"{ci_level:.0f}%"
    mean_appfl = appfl_curves.mean(axis=0)
    mean_ref = reference_curves.mean(axis=0)
    lo_appfl, hi_appfl = bootstrap_ci(appfl_curves, n_boot=n_boot, level=ci_level, seed=1)
    lo_ref, hi_ref = bootstrap_ci(reference_curves, n_boot=n_boot, level=ci_level, seed=2)

    diff_curves = appfl_curves - reference_curves
    lo_diff, hi_diff = bootstrap_ci(diff_curves, n_boot=n_boot, level=ci_level, seed=3)
    mean_diff = diff_curves.mean(axis=0)
    zero_covered = float(np.mean((lo_diff <= 0.0) & (hi_diff >= 0.0)))

    appfl_auc = appfl_curves.mean(axis=1)
    ref_auc = reference_curves.mean(axis=1)
    auc_diff, auc_lo, auc_hi = paired_diff_ci(
        appfl_auc, ref_auc, n_boot=n_boot, level=ci_level, seed=4
    )

    print("\nmean system_best curve")
    print(f"  max |mean APPFL - mean reference|    : {np.abs(mean_appfl - mean_ref).max():.3f}")
    print(f"  mean |mean APPFL - mean reference|   : {np.abs(mean_appfl - mean_ref).mean():.3f}")
    print(f"  paired mean diff CI contains zero    : {zero_covered * 100:.1f}% of rounds")
    print(
        f"  mean per-round AUC diff APPFL-ref    : {auc_diff:+.3f}  "
        f"{ci_label} CI [{auc_lo:+.3f}, {auc_hi:+.3f}]"
    )
    print(
        f"  identical trajectories              : "
        f"{int((np.abs(appfl_curves - reference_curves).max(axis=1) < 1e-9).sum())}"
        f"/{appfl_curves.shape[0]} seeds"
    )
    report = {
        "max_abs_mean_diff": float(np.abs(mean_appfl - mean_ref).max()),
        "mean_abs_mean_diff": float(np.abs(mean_appfl - mean_ref).mean()),
        "pct_rounds_diff_ci_contains_zero": zero_covered * 100.0,
        "mean_per_round_auc_diff": {
            "mean": auc_diff,
            "ci": [auc_lo, auc_hi],
        },
        "identical_trajectories": int(
            (np.abs(appfl_curves - reference_curves).max(axis=1) < 1e-9).sum()
        ),
        "n_seeds": int(appfl_curves.shape[0]),
        "mean_appfl": mean_appfl,
        "mean_reference": mean_ref,
        "mean_diff": mean_diff,
        "diff_ci_low": lo_diff,
        "diff_ci_high": hi_diff,
    }
    return (mean_appfl, mean_ref, lo_appfl, hi_appfl, lo_ref, hi_ref), report


def print_eta_summary(
    appfl: Dict[int, dict],
    reference: Dict[int, dict],
    seeds: Sequence[int],
    n_boot: int,
    ci_level: float,
) -> Optional[dict]:
    """Compare per-step token-fidelity traces."""
    first_appfl_step = appfl[seeds[0]]["steps"][0]
    first_ref_step = reference[seeds[0]]["steps"][0]
    if "mean_token_eta" not in first_appfl_step or "mean_token_eta" not in first_ref_step:
        return None

    eta_appfl = np.array([
        np.nanmean(step_field_matrix(appfl[seed], "mean_token_eta"), axis=1)
        for seed in seeds
    ])
    eta_ref = np.array([
        np.nanmean(step_field_matrix(reference[seed], "mean_token_eta"), axis=1)
        for seed in seeds
    ])
    mean_appfl = eta_appfl.mean(axis=0)
    mean_ref = eta_ref.mean(axis=0)
    final_diff, final_lo, final_hi = paired_diff_ci(
        eta_appfl[:, -1], eta_ref[:, -1], n_boot=n_boot, level=ci_level, seed=5
    )
    auc_diff, auc_lo, auc_hi = paired_diff_ci(
        eta_appfl.mean(axis=1), eta_ref.mean(axis=1),
        n_boot=n_boot, level=ci_level, seed=6
    )
    ci_label = f"{ci_level:.0f}%"
    print(f"\nmean token fidelity eta_bar (final round): "
          f"APPFL {mean_appfl[-1]:.4f}  reference {mean_ref[-1]:.4f}  "
          f"diff {final_diff:+.4f}  {ci_label} CI [{final_lo:+.4f}, {final_hi:+.4f}]")
    print(f"  mean per-round eta_bar diff APPFL-ref: {auc_diff:+.4f}  "
          f"{ci_label} CI [{auc_lo:+.4f}, {auc_hi:+.4f}]")
    print(f"  max |mean eta_bar diff| over rounds : {np.abs(mean_appfl - mean_ref).max():.4f}")
    return {
        "final_round": {
            "appfl": float(mean_appfl[-1]),
            "reference": float(mean_ref[-1]),
            "paired_diff": {
                "mean": final_diff,
                "ci": [final_lo, final_hi],
            },
        },
        "mean_per_round_diff": {
            "mean": auc_diff,
            "ci": [auc_lo, auc_hi],
        },
        "max_abs_mean_diff": float(np.abs(mean_appfl - mean_ref).max()),
        "mean_appfl": mean_appfl,
        "mean_reference": mean_ref,
    }


def print_strict_step_check(
    appfl: Dict[int, dict],
    reference: Dict[int, dict],
    seeds: Sequence[int],
) -> Dict[str, dict]:
    """Report first exact per-step mismatch for each seed."""
    fields = (
        "theta_int",
        "y_raw",
        "score_lam_G",
        "score_gamma_Lambda",
        "n_candidates_scored",
        "n_tokens_in_memory",
    )
    print("\nstrict per-step check")
    report = {}
    for seed in seeds:
        mismatch = first_step_mismatch(appfl[seed], reference[seed], fields, atol=1e-7)
        if mismatch is None:
            print(f"  seed {seed}: exact match on {', '.join(fields)}")
            report[str(seed)] = {"exact": True, "first_mismatch": None}
            continue
        field, round_idx, agent_idx, appfl_value, ref_value = mismatch
        print(
            f"  seed {seed}: first mismatch {field} at round {round_idx}, "
            f"agent {agent_idx}: appfl={appfl_value!r} reference={ref_value!r}"
        )
        report[str(seed)] = {
            "exact": False,
            "first_mismatch": {
                "field": field,
                "round": round_idx,
                "agent": agent_idx,
                "appfl": appfl_value,
                "reference": ref_value,
            },
        }
    return report


def write_plot(
    path: Path,
    arm: str,
    seeds: Sequence[int],
    appfl_curves: np.ndarray,
    reference_curves: np.ndarray,
    ci_level: float,
    plot_arrays: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Write mean-curve and paired-difference plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mean_appfl, mean_ref, lo_appfl, hi_appfl, lo_ref, hi_ref = plot_arrays
    diff_curves = appfl_curves - reference_curves
    mean_diff = diff_curves.mean(axis=0)
    lo_diff, hi_diff = bootstrap_ci(diff_curves, level=ci_level, seed=3)
    rounds = np.arange(appfl_curves.shape[1])

    fig, (ax, diff_ax) = plt.subplots(
        2, 1, figsize=(7, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )
    for mean, lo, hi, label, color in (
        (mean_ref, lo_ref, hi_ref, "reference (scientific_discovery)", "#444444"),
        (mean_appfl, lo_appfl, hi_appfl, "APPFL decentralized", "#c2410c"),
    ):
        ax.plot(rounds, mean, label=label, color=color, lw=1.8)
        ax.fill_between(rounds, lo, hi, color=color, alpha=0.18, lw=0)
    ax.set_ylabel("system best yield (%)")
    ax.set_title(
        f"ADKO {arm}, LLM off -- {len(seeds)} paired seeds "
        f"({ci_level:.0f}% bootstrap CI)"
    )
    ax.legend(loc="lower right", frameon=False)
    ax.grid(alpha=0.25, lw=0.5)

    diff_ax.axhline(0.0, color="#666666", lw=0.8)
    diff_ax.plot(rounds, mean_diff, color="#2563eb", lw=1.5)
    diff_ax.fill_between(rounds, lo_diff, hi_diff, color="#2563eb", alpha=0.18, lw=0)
    diff_ax.set_xlabel("round")
    diff_ax.set_ylabel("APPFL - ref")
    diff_ax.grid(alpha=0.25, lw=0.5)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--appfl", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--arm", choices=["IID", "HET"], required=True)
    parser.add_argument("--plot", type=Path, default=None,
                        help="optional PNG of the two mean curves with CIs")
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT,
                        help="bootstrap resamples for confidence intervals")
    parser.add_argument("--ci", type=float, default=DEFAULT_CI,
                        help="confidence interval level, e.g. 95")
    parser.add_argument("--strict-steps", action="store_true",
                        help="also report first per-step mismatch in key implementation fields")
    parser.add_argument("--report-json", type=Path, default=None,
                        help="optional JSON file containing the comparison summary")
    args = parser.parse_args()

    appfl = load_runs(args.appfl, args.arm)
    reference = load_runs(args.reference, args.arm)
    seeds = sorted(set(appfl) & set(reference))
    if not seeds:
        raise SystemExit("no overlapping seeds between the two directories")
    only_a, only_r = sorted(set(appfl) - set(seeds)), sorted(set(reference) - set(seeds))
    if only_a or only_r:
        print(f"note: comparing {len(seeds)} shared seeds "
              f"(appfl-only {len(only_a)}, reference-only {len(only_r)})")

    appfl_curves = system_best_matrix(appfl, seeds)
    reference_curves = system_best_matrix(reference, seeds)

    ci_label = f"{args.ci:.0f}%"
    print(f"\n=== {args.arm} / ADKO / LLM off -- {len(seeds)} paired seeds ===")
    print(f"    appfl     {args.appfl}")
    print(f"    reference {args.reference}\n")

    report = {
        "arm": args.arm,
        "appfl_dir": str(args.appfl),
        "reference_dir": str(args.reference),
        "n_paired_seeds": len(seeds),
        "paired_seeds": seeds,
        "n_boot": args.n_boot,
        "ci": args.ci,
        "config_mismatches": print_config_warnings(
            appfl[seeds[0]]["config"], reference[seeds[0]]["config"]
        ),
    }
    report["warmup"] = print_warmup_check(appfl, reference, seeds)
    report["thresholds"] = print_threshold_table(
        appfl_curves, reference_curves, args.n_boot, args.ci
    )
    plot_arrays, report["system_best"] = curve_summary(
        appfl_curves, reference_curves, args.n_boot, args.ci
    )
    eta_report = print_eta_summary(appfl, reference, seeds, args.n_boot, args.ci)
    if eta_report is not None:
        report["eta_bar"] = eta_report
    if args.strict_steps:
        report["strict_steps"] = print_strict_step_check(appfl, reference, seeds)

    if args.plot:
        write_plot(
            args.plot,
            args.arm,
            seeds,
            appfl_curves,
            reference_curves,
            args.ci,
            plot_arrays,
        )
        print(f"\nwrote {args.plot}")

    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(jsonable(report), indent=2) + "\n")
        print(f"wrote {args.report_json}")


if __name__ == "__main__":
    main()
