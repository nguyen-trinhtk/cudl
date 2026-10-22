#!/usr/bin/env python3
"""Plot benchmark results (JSON lines written with --json): one panel per op, a bar group per shape.

    build/bench/bench_reduce --json results.jsonl
    scripts/plot_bench.py results.jsonl [-o bench.png] [--metric gbps|pct|ms]
                          [--op REGEX] [--algo REGEX] [--load REGEX]

Bar color is the kernel's algo, solid for scalar loads and hatched for vec4, so both axes can be read
off one panel. The dashed line is the best cudaMemcpy throughput in the file, the practical ceiling for
memory-bound ops. --op/--algo/--load must match the whole axis name, like the benchmark binaries' flags.
If a file holds several results for the same key (e.g. appended runs), the last one wins. Needs matplotlib.
"""

import argparse
import json
import math
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# Same categorical slots as plot_drift.py, assigned to algos in order of first appearance in the file.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3de", "#ffffff"
BASELINE = "cudaMemcpy"

METRICS = {
    "gbps": ("GB/s", lambda r: r["gbps"]),
    "pct": ("% of peak bandwidth", lambda r: r["pct_peak_bw"]),
    "ms": ("median time per launch (ms)", lambda r: r["median_ms"]),
}


def load(path):
    rows = {}
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                sys.exit(f"{path}:{lineno}: {e}")
            rows[(r["gpu"], r["op"], r["algo"], r["load"], r["shape"])] = r
    return list(rows.values())


def shape_key(shape):
    dims = [int(d) for d in shape.split("x")]
    return math.prod(dims), dims


def shape_label(shape):
    """1048576 -> 1M; 2-D shapes stay as written."""
    if "x" in shape:
        return shape
    n = int(shape)
    for unit, scale in (("G", 1 << 30), ("M", 1 << 20), ("K", 1 << 10)):
        if n >= scale and n % scale == 0:
            return f"{n // scale}{unit}"
    return shape


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl")
    ap.add_argument("-o", "--out", default="bench.png")
    ap.add_argument("--metric", choices=METRICS, default="gbps")
    ap.add_argument("--op", default=".*")
    ap.add_argument("--algo", default=".*")
    ap.add_argument("--load", default=".*")
    ap.add_argument("--cols", type=int, default=3, help="panels per row (default 3)")
    args = ap.parse_args()

    rows = load(args.jsonl)
    if len({r["gpu"] for r in rows}) > 1:
        sys.exit("results from more than one GPU; split the file first")

    # Colors are fixed before filtering, so a filter never repaints the algos that remain.
    algos = list(dict.fromkeys(r["algo"] for r in rows if r["load"] != "-"))
    color = {a: COLORS[i] if i < len(COLORS) else MUTED for i, a in enumerate(algos)}
    label, value = METRICS[args.metric]

    baselines = [r for r in rows if r["algo"] == BASELINE]
    ceiling = max((value(r) for r in baselines), default=None) if args.metric != "ms" else None

    keep = [r for r in rows if r["load"] != "-" and re.fullmatch(args.op, r["op"])
            and re.fullmatch(args.algo, r["algo"]) and re.fullmatch(args.load, r["load"])]
    if not keep:
        sys.exit("no results match the filters")
    ops = list(dict.fromkeys(r["op"] for r in keep))

    cols = min(args.cols, len(ops))
    nrows = math.ceil(len(ops) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(max(5.2 * cols, 8), 3.4 * nrows + 0.8), dpi=150, squeeze=False)
    fig.patch.set_facecolor(SURFACE)

    for ax, op in zip(axes.flat, ops):
        op_rows = [r for r in keep if r["op"] == op]
        shapes = sorted({r["shape"] for r in op_rows}, key=shape_key)
        variants = list(dict.fromkeys((r["algo"], r["load"]) for r in op_rows))
        by_key = {(r["algo"], r["load"], r["shape"]): r for r in op_rows}

        slot = 0.8 / len(variants)
        for i, (algo, ld) in enumerate(variants):
            xs, ys = [], []
            for j, s in enumerate(shapes):
                r = by_key.get((algo, ld, s))
                if r:
                    xs.append(j - 0.4 + slot * (i + 0.5))
                    ys.append(value(r))
            ax.bar(xs, ys, width=slot, color=color[algo], edgecolor=SURFACE, linewidth=1,
                   hatch="////" if ld == "vec4" else None)

        if ceiling is not None:
            ax.axhline(ceiling, color=MUTED, linestyle="--", linewidth=1, zorder=3)
        ax.set_title(op, color=INK, loc="left", fontsize=11)
        ax.set_xticks(range(len(shapes)))
        ax.set_xticklabels([shape_label(s) for s in shapes], rotation=30 if len(shapes) > 4 else 0,
                           ha="right" if len(shapes) > 4 else "center", fontsize=8)
        ax.grid(True, axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=MUTED, length=0)
    for ax in axes.flat[len(ops):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel(label, color=MUTED)

    # Shared legend: one entry per algo shown, then the load encoding and the ceiling.
    shown = list(dict.fromkeys(r["algo"] for r in keep))
    handles = [Patch(facecolor=color[a], label=a or "(no algo)") for a in shown]
    loads = {r["load"] for r in keep}
    if "scalar" in loads:
        handles.append(Patch(facecolor=MUTED, edgecolor=SURFACE, label="scalar"))
    if "vec4" in loads:
        handles.append(Patch(facecolor=MUTED, edgecolor=SURFACE, hatch="////", label="vec4"))
    if ceiling is not None:
        handles.append(plt.Line2D([], [], color=MUTED, linestyle="--", label=f"best {BASELINE}"))

    gpu = rows[0]["gpu"]
    sha = rows[0].get("git_sha", "?")
    flush = " · cold L2" if rows[0].get("flush_l2") else ""
    # Title on the first line, legend below it, wrapped to the figure width (~1.3 in per entry).
    width, height = fig.get_size_inches()
    ncol = max(1, min(len(handles), int(width / 1.3)))
    header = 0.2 + 0.28 * math.ceil(len(handles) / ncol)  # inches
    fig.suptitle(f"{label} · {gpu} · {sha}{flush}", color=INK, x=0.01, y=1 - 0.08 / height, ha="left",
                 va="top", fontsize=12)
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, 1 - 0.38 / height), ncol=ncol,
               frameon=False, labelcolor=INK, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 1 - header / height))
    fig.savefig(args.out, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
