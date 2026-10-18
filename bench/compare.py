#!/usr/bin/env python3
"""Diff two benchmark runs (JSON lines written with --json).

    bench/compare.py old.jsonl new.jsonl [--threshold 5] [--op REGEX] [--algo REGEX] [--load REGEX]
                                         [--filter REGEX]

Rows are matched on (gpu, op, algo, load, shape). --op/--algo/--load must match the whole axis
name, like the benchmark binaries' flags; --filter searches "op/variant/shape". If a file holds several results for the same
key (e.g. appended runs), the last one wins. Prints old and new median time and the speedup;
changes beyond --threshold percent are marked as faster (+) or slower (-). Exits with 1 when
any matched row got slower than the threshold, so it can gate CI.
"""

import argparse
import json
import re
import sys


def axes(r):
    """(algo, load) of a result; derived from the variant name for runs that predate those fields."""
    if "algo" in r and "load" in r:
        return r["algo"], r["load"]
    v = r["variant"]
    if v == "cudaMemcpy":  # baseline row
        return v, "-"
    if v in ("naive", "scalar"):
        return "", "scalar"
    if v == "vec4":
        return "", "vec4"
    if v.endswith("_vec4"):
        return v[: -len("_vec4")], "vec4"
    return v, "scalar"


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
            algo, ld = axes(r)
            rows[(r.get("gpu", "?"), r["op"], algo, ld, r["shape"])] = r
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--threshold", type=float, default=5.0, help="percent change to flag (default 5)")
    ap.add_argument("--op", help="only rows whose op fully matches this regex")
    ap.add_argument("--algo", help="only rows whose algo fully matches this regex")
    ap.add_argument("--load", help="only rows whose load (scalar/vec4) fully matches this regex")
    ap.add_argument("--filter", help="only rows whose op/variant/shape contains a match for this regex")
    args = ap.parse_args()

    old, new = load(args.old), load(args.new)
    axis_res = [(i, re.compile(p)) for i, p in ((1, args.op), (2, args.algo), (3, args.load)) if p is not None]
    pattern = re.compile(args.filter) if args.filter else None

    def keep(k):
        if any(not rx.fullmatch(k[i]) for i, rx in axis_res):
            return False
        return not pattern or pattern.search(f"{k[1]}/{new[k]['variant']}/{k[4]}")

    keys = [k for k in new if k in old and keep(k)]
    if not keys:
        sys.exit("no matching rows between the two runs (different GPUs, ops or shapes?)")

    shas = {r.get("git_sha", "?") for r in old.values()}, {r.get("git_sha", "?") for r in new.values()}
    print(f"old: {args.old} ({', '.join(sorted(shas[0]))})")
    print(f"new: {args.new} ({', '.join(sorted(shas[1]))})\n")
    print(f"{'op':<10} {'algo':<10} {'load':<6} {'shape':<14} {'old_ms':>10} {'new_ms':>10} {'speedup':>8} {'GB/s':>9}")

    slower = 0
    for key in sorted(keys, key=lambda k: k[1:]):
        o, n = old[key], new[key]
        speedup = o["median_ms"] / n["median_ms"] if n["median_ms"] > 0 else float("inf")
        change = (speedup - 1.0) * 100.0
        mark = ""
        if change > args.threshold:
            mark = " +"
        elif change < -args.threshold:
            mark = " -"
            slower += 1
        _, op, algo, ld, shape = key
        print(f"{op:<10} {algo or '-':<10} {ld:<6} {shape:<14} {o['median_ms']:>10.4f} {n['median_ms']:>10.4f} "
              f"{speedup:>7.2f}x {n.get('gbps', 0):>9.1f}{mark}")

    only_old = len([k for k in old if k not in new])
    only_new = len([k for k in new if k not in old])
    print(f"\n{len(keys)} matched, {slower} slower than {args.threshold:g}%"
          + (f", {only_old} only in old" if only_old else "")
          + (f", {only_new} only in new" if only_new else ""))
    return 1 if slower else 0


if __name__ == "__main__":
    sys.exit(main())
