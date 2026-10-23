"""Turn a results tree of ``run.json`` files into the table the screen is read from.

Two things it does that a mean and a standard deviation do not:

* **pairs by seed.** Arms share their seeds, their shared initial points and their graph, so
  the comparison that matters is the per-seed difference against the baseline arm, and the
  test is a paired Wilcoxon signed-rank -- the same test the v2 findings use, so numbers from
  the two methods can be read side by side.
* **holds the claim to every seed.** An arm is reported as winning only if its mean is better
  *and* it wins at every seed -- a strict pre-registration rule, applied here so that a gain
  carried by one lucky seed does not count. That column is the one worth believing.

    python examples/decentralized/adko_h/report.py $ADKO_RESULTS/zaleplon_mpo
    python examples/decentralized/adko_h/report.py $ADKO_RESULTS/* --metric diverse_hits
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

#: Endpoints worth a column, in the order the paper reads them.
METRICS = (
    "distinct_families",
    "diverse_hits",
    "n_circles",
    "hull_volume",
    "auc_top10",
    "best",
    "distinct_keys",
)


def load(roots: Sequence[str]) -> Dict[str, Dict[int, dict]]:
    """``{arm: {seed: record}}`` from every ``run.json`` under the given roots."""
    out: Dict[str, Dict[int, dict]] = {}
    for root in roots:
        for path in sorted(Path(root).rglob("run.json")):
            try:
                rec = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            arm = rec.get("label") or path.parent.parent.name
            seed = int(rec.get("config", {}).get("seed", -1))
            out.setdefault(arm, {})[seed] = rec
    return out


def wilcoxon(diffs: Sequence[float]) -> Optional[float]:
    """Two-sided paired Wilcoxon signed-rank p-value; exact for the sizes a screen has.

    Ties and zeros are dropped as in the standard treatment. ``None`` when nothing is left,
    which is itself worth printing: it means the arms were identical on every seed.
    """
    d = [x for x in diffs if x != 0 and not math.isnan(x)]
    n = len(d)
    if n == 0:
        return None
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:  # average the ranks of equal magnitudes
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w = min(sum(r for r, x in zip(ranks, d) if x > 0), sum(r for r, x in zip(ranks, d) if x < 0))
    if n > 20:  # normal approximation, as scipy does
        mean = n * (n + 1) / 4.0
        sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
        z = (w - mean) / sd if sd else 0.0
        return max(min(2.0 * 0.5 * math.erfc(abs(z) / math.sqrt(2.0)), 1.0), 0.0)
    count = 0
    total = 1 << n
    for mask in range(total):  # exact: every sign assignment
        s = sum(ranks[i] for i in range(n) if mask >> i & 1)
        if min(s, sum(ranks) - s) <= w + 1e-12:
            count += 1
    return min(count / total, 1.0)


def column(records: Dict[int, dict], metric: str) -> Dict[int, float]:
    return {
        seed: float(rec.get("endpoints", {}).get(metric, float("nan")))
        for seed, rec in records.items()
    }


def table(data: Dict[str, Dict[int, dict]], metric: str, baseline: str) -> List[str]:
    base = column(data.get(baseline, {}), metric)
    rows = [
        f"{'arm':20s} {'n':>3s} {'mean':>9s} {'sd':>8s} {'vs ' + baseline:>12s} "
        f"{'p':>7s} {'every seed':>10s}"
    ]
    for arm in sorted(data):
        col = column(data[arm], metric)
        vals = [v for v in col.values() if not math.isnan(v)]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        sd = (sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5
        shared = sorted(s for s in set(col) & set(base) if not math.isnan(col[s] - base[s]))
        diffs = [col[s] - base[s] for s in shared]
        if arm == baseline or not diffs:
            rows.append(f"{arm:20s} {len(vals):3d} {mean:9.3f} {sd:8.3f} {'--':>12s} {'--':>7s} {'--':>10s}")
            continue
        p = wilcoxon(diffs)
        delta = sum(diffs) / len(diffs)
        every = "yes" if all(x > 0 for x in diffs) else ("no" if any(x > 0 for x in diffs) else "worse")
        rows.append(
            f"{arm:20s} {len(vals):3d} {mean:9.3f} {sd:8.3f} {delta:+12.3f} "
            f"{('n/a' if p is None else f'{p:.3f}'):>7s} {every:>10s}"
        )
    return rows


def cost(data: Dict[str, Dict[int, dict]]) -> List[str]:
    """What each arm spent: model calls, tokens, token size, and planner calibration."""
    rows = [f"{'arm':20s} {'designs':>8s} {'fallback':>8s} {'in-tok':>9s} {'out-tok':>9s} {'$/run':>7s} {'bits/tok':>9s} {'brier':>7s} {'skill':>7s} {'pass':>9s}"]
    for arm in sorted(data):
        recs = list(data[arm].values())
        g = lambda k, d=0.0: sum(float((r.get("planner") or {}).get(k, d) or d) for r in recs) / len(recs)
        c = [r.get("campaigns", {}) for r in recs]
        cal = [x.get("calibration", {}) for x in c]
        br = [x["brier"] for x in cal if x.get("brier") is not None]
        sk = [x["skill"] for x in cal if x.get("skill") is not None]
        n = sum(int(x.get("n", 0)) for x in c)
        passed = sum(int(x.get("passed", 0)) for x in c)
        fallback = g("errors") + g("unusable")
        rows.append(
            f"{arm:20s} {g('designs'):8.0f} {fallback:8.1f} {g('input_tokens'):9.0f} "
            f"{g('output_tokens'):9.0f} {g('cost_usd'):7.2f} "
            f"{sum(float(r.get('bits_per_token', 0)) for r in recs) / len(recs):9.0f} "
            f"{(sum(br) / len(br) if br else float('nan')):7.3f} "
            f"{(sum(sk) / len(sk) if sk else float('nan')):7.3f} "
            f"{(f'{passed}/{n}' if n else '--'):>9s}"
        )
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("roots", nargs="+", help="results directories to walk")
    p.add_argument("--metric", default=None, help="one endpoint; default prints all of them")
    p.add_argument("--baseline", default="adko", help="arm every other arm is paired against")
    args = p.parse_args(argv)

    data = load(args.roots)
    if not data:
        print("no run.json found")
        return 1
    seeds = sorted({s for arm in data.values() for s in arm})
    print(f"arms: {', '.join(sorted(data))}")
    print(f"seeds: {seeds[0]}-{seeds[-1]} ({len(seeds)})\n")
    for metric in [args.metric] if args.metric else METRICS:
        print(f"== {metric}")
        print("\n".join(table(data, metric, args.baseline)))
        print()
    print("== cost and calibration (per run; fallback = designs that failed and became random)")
    print("\n".join(cost(data)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
