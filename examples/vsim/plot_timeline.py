"""Generate a virtual-time timeline diagram from an async vsim run.

Shows per-client training & communication intervals, global model
aggregation points, and global eval markers on a shared time axis.

The input is the history JSON written by ``run_vsim.py --save_history``. A
``vsim.log`` is also accepted, for runs made without that flag, and its event
table is parsed on a best-effort basis.

Usage:
    python vsim/plot_timeline.py <history.json | vsim.log> [--out <output.png>] [--max_vt <seconds>]
"""

import os
import re
import json
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Rows of the event table that AsyncSimDriver logs, e.g.
#   12.3456      START    Client3     1.2345     1.1000     0.1345
#   13.5801       DONE    Client3                                           7          2    93.1000
# DONE rows leave the duration columns blank, so the fields after the client are
# read according to the event, not by column position.
_ROW_RE = re.compile(r"([\d.]+)\s+(START|DONE)\s+(\S+)(.*)$")
_EVAL_RE = re.compile(r"global eval @ epoch (\d+): val_acc=(\S+)\s+val_loss=(\S+)")


def load_history(path):
    """
    Read the async history JSON written by ``run_vsim.py --save_history``.

    Each record is one client update and becomes a start (dispatch time and the
    compute/comm split), a completion, and, when global evaluation ran at that
    arrival, an eval point.
    """
    with open(path) as f:
        history = json.load(f)
    if any("cid" not in r or "dispatch_time" not in r for r in history):
        raise SystemExit(
            f"{path} is not an async history (records lack `cid` and "
            "`dispatch_time`); plot_timeline.py only plots async runs."
        )

    starts, dones, evals = [], [], []
    for r in history:
        starts.append(
            {
                "vt": r["dispatch_time"],
                "client": r["cid"],
                "dur": r["duration"],
                # None when the record carries no split; drawn as a single bar.
                "compute": r.get("compute_time"),
                "comm": r.get("comm_time"),
            }
        )
        dones.append(
            {
                "vt": r["vtime"],
                "client": r["cid"],
                "epoch": r["epoch"],
                "staleness": r["staleness"],
            }
        )
        if isinstance(r.get("global_val_accuracy"), (int, float)):
            evals.append(
                {
                    "vt": r["vtime"],
                    "epoch": r["epoch"],
                    "acc": r["global_val_accuracy"],
                    "loss": r.get("global_val_loss"),
                }
            )
    return starts, dones, evals


def parse_log(path):
    """
    Best-effort parse of the async event table in a vsim log.

    Prefer the history JSON: the log rounds values to four decimals, and a client
    id containing whitespace breaks its rows. A global eval line carries no
    virtual time, so it takes the time of the DONE row logged just before it,
    which is the arrival that triggered it.
    """
    starts, dones, evals = [], [], []
    with open(path) as f:
        for line in f:
            m = _ROW_RE.search(line)
            if m:
                fields = m.group(4).split()
                try:
                    vt = float(m.group(1))
                    if m.group(2) == "START":
                        dur, compute, comm = (float(x) for x in fields[:3])
                        starts.append(
                            {
                                "vt": vt,
                                "client": m.group(3),
                                "dur": dur,
                                "compute": compute,
                                "comm": comm,
                            }
                        )
                    else:
                        dones.append(
                            {
                                "vt": vt,
                                "client": m.group(3),
                                "epoch": int(fields[0]),
                                "staleness": int(fields[1]),
                            }
                        )
                except (ValueError, IndexError):
                    pass  # malformed or truncated row
                continue
            m = _EVAL_RE.search(line)
            if m and dones:
                evals.append(
                    {
                        "vt": dones[-1]["vt"],
                        "epoch": int(m.group(1)),
                        "acc": float(m.group(2)),
                        "loss": float(m.group(3)),
                    }
                )
    return starts, dones, evals


def load_events(path):
    """
    Load ``(starts, dones, evals)`` from a history JSON or a vsim log.

    A ``.json`` path is read as the history written by
    ``run_vsim.py --save_history``; anything else is parsed as a vsim log.
    """
    if path.lower().endswith(".json"):
        return load_history(path)
    return parse_log(path)


def events_until(events, max_vt):
    """Keep the ``(starts, dones, evals)`` at or before ``max_vt``, if it is set."""
    if not max_vt:
        return events
    return tuple([e for e in group if e["vt"] <= max_vt] for group in events)


def plot_timeline(starts, dones, evals, out_path, max_vt=None):
    clients = sorted(
        {s["client"] for s in starts} | {d["client"] for d in dones},
        key=lambda c: int(re.sub(r"\D", "", c) or 0),
    )
    cid_to_y = {c: i for i, c in enumerate(clients)}

    fig, (ax_timeline, ax_acc) = plt.subplots(
        2,
        1,
        figsize=(16, 8),
        height_ratios=[3, 1],
        sharex=True,
        gridspec_kw={"hspace": 0.08},
    )

    bar_h = 0.7
    has_split = has_unsplit = False

    for s in starts:
        y = cid_to_y[s["client"]]
        vt_start = s["vt"]

        if s["compute"] is None or s["comm"] is None:
            # no compute/comm split recorded: one bar for the whole duration
            ax_timeline.barh(
                y,
                s["dur"],
                left=vt_start,
                height=bar_h,
                color="#8C8C8C",
                edgecolor="white",
                linewidth=0.3,
            )
            has_unsplit = True
            continue
        has_split = True
        compute_end = vt_start + s["compute"]

        # compute bar
        ax_timeline.barh(
            y,
            s["compute"],
            left=vt_start,
            height=bar_h,
            color="#4C72B0",
            edgecolor="white",
            linewidth=0.3,
        )
        # comm bar
        ax_timeline.barh(
            y,
            s["comm"],
            left=compute_end,
            height=bar_h,
            color="#DD8452",
            edgecolor="white",
            linewidth=0.3,
        )

    # aggregation points (DONE events)
    for d in dones:
        y = cid_to_y[d["client"]]
        ax_timeline.plot(
            d["vt"], y, marker="|", color="black", markersize=8, markeredgewidth=1.2
        )

    # global eval lines
    for e in evals:
        ax_timeline.axvline(
            e["vt"], color="#55A868", alpha=0.5, linewidth=1, linestyle="--"
        )

    ax_timeline.set_yticks(range(len(clients)))
    ax_timeline.set_yticklabels(clients, fontsize=9)
    ax_timeline.set_ylabel("Client")
    ax_timeline.set_title("Virtual-Time Async FL Timeline (single GPU simulation)")
    ax_timeline.invert_yaxis()

    legend_patches = []
    if has_split:
        legend_patches += [
            mpatches.Patch(color="#4C72B0", label="Compute"),
            mpatches.Patch(color="#DD8452", label="Communication"),
        ]
    if has_unsplit:
        legend_patches.append(
            mpatches.Patch(color="#8C8C8C", label="Compute + comm (no split)")
        )
    legend_patches += [
        plt.Line2D(
            [0],
            [0],
            color="black",
            marker="|",
            linestyle="None",
            markersize=8,
            label="Aggregation",
        ),
        plt.Line2D([0], [0], color="#55A868", linestyle="--", label="Global Eval"),
    ]
    ax_timeline.legend(handles=legend_patches, loc="upper right", fontsize=8)

    # accuracy curve
    ax_acc.set_xlabel("Virtual Time (seconds)")
    if evals:
        vts = [e["vt"] for e in evals]
        accs = [e["acc"] for e in evals]
        ax_acc.plot(vts, accs, "-o", color="#4C72B0", markersize=3, linewidth=1.5)
        ax_acc.set_ylabel("Global Accuracy (%)")
        ax_acc.grid(True, alpha=0.3)
        ax_acc.set_ylim(0, max(accs) * 1.15)
    else:
        ax_acc.text(
            0.5,
            0.5,
            "no global eval in this run "
            "(needs --eval_every and a server val_data_configs)",
            transform=ax_acc.transAxes,
            ha="center",
            va="center",
            color="gray",
        )
        ax_acc.set_yticks([])

    if max_vt:
        ax_timeline.set_xlim(-0.2, max_vt)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        help="history JSON from run_vsim.py --save_history, "
        "or a vsim log (parsed best-effort)",
    )
    parser.add_argument("--out", default=None, help="output image path")
    parser.add_argument(
        "--max_vt",
        type=float,
        default=None,
        help="only show up to this virtual time (for zoomed view)",
    )
    args = parser.parse_args()

    if args.out is None:
        args.out = os.path.splitext(args.path)[0] + "_timeline.png"

    events = load_events(args.path)
    starts, dones, evals = events
    print(
        f"Loaded: {len(starts)} starts, {len(dones)} completions, {len(evals)} global evals"
    )
    if not starts:
        raise SystemExit(
            f"No async client events found in {args.path}. Expected the history "
            "JSON from run_vsim.py --save_history, or the log of an async run."
        )

    plot_timeline(*events_until(events, args.max_vt), args.out, args.max_vt)

    # also generate a zoomed-in view of the first few seconds
    if args.max_vt is None:
        zoom_vt = 5.0
        root, ext = os.path.splitext(args.out)
        zoom_out = root + "_zoom" + (ext or ".png")
        plot_timeline(*events_until(events, zoom_vt), zoom_out, zoom_vt)


if __name__ == "__main__":
    main()
