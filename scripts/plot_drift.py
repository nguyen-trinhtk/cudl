#!/usr/bin/env python3
"""Plot the summation drift sweep: relative error vs N for each sum variant (log-log).

    CUDL_DRIFT_CSV=drift.csv build/tests/cudl_tests --gtest_filter='ReduceDrift.*'
    scripts/plot_drift.py drift.csv [-o drift.png]

Needs matplotlib. Errors of exactly 0 are drawn at the bottom of the axis.
"""

import argparse
import csv
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots 1-5 in fixed order; markers double-encode identity for colorblind readers.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("-o", "--out", default="drift.png")
    args = ap.parse_args()

    with open(args.csv) as f:
        rows = list(csv.DictReader(f))
    ns = [int(r["n"]) for r in rows]
    series = [k for k in rows[0] if k != "n"]
    values = {s: [float(r[s]) for r in rows] for s in series}
    floor = min((v for vs in values.values() for v in vs if v > 0), default=1e-9) / 4

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ends = {}
    for i, s in enumerate(series):
        ys = [max(v, floor) for v in values[s]]
        ax.plot(ns, ys, color=COLORS[i % len(COLORS)], marker=MARKERS[i % len(MARKERS)], markersize=6,
                linewidth=2, label=s)
        ends[s] = math.log10(ys[-1])

    # End-of-line labels, nudged apart (in log space) so converging lines stay readable.
    all_logs = [math.log10(max(v, floor)) for vs in values.values() for v in vs]
    gap = 0.05 * max(max(all_logs) - min(all_logs), 1.0)
    placed = []
    for s, y in sorted(ends.items(), key=lambda kv: kv[1]):
        if placed and y - placed[-1][1] < gap:
            y = placed[-1][1] + gap
        placed.append((s, y))
    for s, y in placed:
        ax.annotate(s, (ns[-1], 10**ends[s]), xytext=(ns[-1] * 1.3, 10**y), textcoords="data", va="center",
                    fontsize=9, color=INK)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("N (elements summed, uniform [0, 1))", color=MUTED)
    ax.set_ylabel("relative error vs double reference", color=MUTED)
    ax.set_title("fp32 summation drift", color=INK, loc="left")
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, loc="upper left", labelcolor=INK)
    ax.set_xlim(right=ns[-1] * 4)  # room for the end labels
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
