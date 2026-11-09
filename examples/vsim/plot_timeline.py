"""Generate a virtual-time timeline diagram from vsim log.

Shows per-client training & communication intervals, global model
aggregation points, and global eval markers on a shared time axis.

Usage:
    python vsim/plot_timeline.py <log_file> [--out <output.png>] [--max_vt <seconds>]
"""

import argparse
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def parse_log(path, max_vt=None):
    start_re = re.compile(
        r"\[vt=\s*([\d.]+)\]\s+START\s+(\S+)\s+dur=\s*([\d.]+)\s+"
        r"\(compute=([\d.]+)\+comm=([\d.]+)\)"
    )
    done_re = re.compile(
        r"\[vt=\s*([\d.]+)\]\s+DONE\s+(\S+)\s+epoch=\s*(\d+)\s+staleness=\s*(\d+)"
    )
    global_re = re.compile(
        r"\[vt=\s*([\d.]+)\]\s+GLOBAL\s+epoch=\s*(\d+)\s+"
        r"global_val_acc=([\d.]+)\s+global_val_loss=([\d.]+)"
    )

    starts = []
    dones = []
    evals = []

    with open(path) as f:
        for line in f:
            if "INFO:" in line:
                continue
            m = start_re.search(line)
            if m:
                vt = float(m.group(1))
                if max_vt and vt > max_vt:
                    continue
                starts.append(
                    {
                        "vt": vt,
                        "client": m.group(2),
                        "dur": float(m.group(3)),
                        "compute": float(m.group(4)),
                        "comm": float(m.group(5)),
                    }
                )
                continue
            m = done_re.search(line)
            if m:
                vt = float(m.group(1))
                if max_vt and vt > max_vt:
                    continue
                dones.append(
                    {
                        "vt": vt,
                        "client": m.group(2),
                        "epoch": int(m.group(3)),
                        "staleness": int(m.group(4)),
                    }
                )
                continue
            m = global_re.search(line)
            if m:
                vt = float(m.group(1))
                if max_vt and vt > max_vt:
                    continue
                evals.append(
                    {
                        "vt": vt,
                        "epoch": int(m.group(2)),
                        "acc": float(m.group(3)),
                        "loss": float(m.group(4)),
                    }
                )
    return starts, dones, evals


def plot_timeline(starts, dones, evals, out_path, max_vt=None):
    clients = sorted(
        {s["client"] for s in starts}, key=lambda c: int(re.sub(r"\D", "", c) or 0)
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

    for s in starts:
        y = cid_to_y[s["client"]]
        vt_start = s["vt"]
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

    legend_patches = [
        mpatches.Patch(color="#4C72B0", label="Compute"),
        mpatches.Patch(color="#DD8452", label="Communication"),
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
    if evals:
        vts = [e["vt"] for e in evals]
        accs = [e["acc"] for e in evals]
        ax_acc.plot(vts, accs, "-o", color="#4C72B0", markersize=3, linewidth=1.5)
        ax_acc.set_ylabel("Global Accuracy (%)")
        ax_acc.set_xlabel("Virtual Time (seconds)")
        ax_acc.grid(True, alpha=0.3)
        ax_acc.set_ylim(0, max(accs) * 1.15)

    if max_vt:
        ax_timeline.set_xlim(-0.2, max_vt)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", help="vsim log file")
    parser.add_argument("--out", default=None, help="output image path")
    parser.add_argument(
        "--max_vt",
        type=float,
        default=None,
        help="only show up to this virtual time (for zoomed view)",
    )
    args = parser.parse_args()

    if args.out is None:
        args.out = args.log.rsplit(".", 1)[0] + "_timeline.png"

    starts, dones, evals = parse_log(args.log, args.max_vt)
    print(
        f"Parsed: {len(starts)} starts, {len(dones)} completions, {len(evals)} global evals"
    )

    plot_timeline(starts, dones, evals, args.out, args.max_vt)

    # also generate a zoomed-in view of the first few seconds
    if args.max_vt is None and starts:
        zoom_vt = 5.0
        s2, d2, e2 = parse_log(args.log, zoom_vt)
        zoom_out = args.out.rsplit(".", 1)[0] + "_zoom.png"
        plot_timeline(s2, d2, e2, zoom_out, zoom_vt)


if __name__ == "__main__":
    main()
