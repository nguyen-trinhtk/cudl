#!/bin/bash
# Run every benchmark binary on one GPU and collect the results in one JSON-lines file.
# Extra arguments go to each benchmark.
#
#   sbatch scripts/slurm/bench.sbatch
#   sbatch scripts/slurm/bench.sbatch --load vec4 --iters 200
#   bench/compare.py results/bench-<old>.jsonl results/bench-<new>.jsonl
#SBATCH -J cudl-bench
#SBATCH -A spiralfreeprj
#SBATCH --qos=free
#SBATCH -p gpu-sm
#SBATCH --gres=gpu:v100:1
#SBATCH -c 4
#SBATCH -t 01:00:00
set -euo pipefail
# Slurm runs a copy of this script, so find the original (as submitted) to locate env.sh.
source "$(dirname "$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/^ *Command=\([^ ]*\).*/\1/p')")/env.sh"

out=results/bench-$(git rev-parse --short HEAD)-${SLURM_JOB_ID}.jsonl
for b in copy elementwise reduce nn; do
    build/bench/bench_$b --json "$out" "$@"
done
echo "results: $out"
