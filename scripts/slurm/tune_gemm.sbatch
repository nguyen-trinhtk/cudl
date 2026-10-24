#!/bin/bash
# Run the smem_2d vec4 GEMM tile size sweep (bench_gemm_tune) and save the results to a .jsonl file.
# Extra arguments go to bench_gemm_tune (default size 4096).
#
#   sbatch scripts/slurm/tune_gemm.sbatch
#   sbatch scripts/slurm/tune_gemm.sbatch --size 2048 --algo 'smem_2d_128x.*'
#SBATCH -J cudl-tune-gemm
#SBATCH -A spiralfreeprj
#SBATCH --qos=free
#SBATCH -p gpu-sm
#SBATCH --gres=gpu:v100:1
#SBATCH -c 4
#SBATCH -t 00:30:00
set -euo pipefail
# Slurm runs a copy of this script, so find the original (as submitted) to locate env.sh.
source "$(dirname "$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/^ *Command=\([^ ]*\).*/\1/p')")/env.sh"

out=results/tune-gemm-$(git rev-parse --short HEAD)-${SLURM_JOB_ID}.jsonl
build/bench/bench_gemm_tune --json "$out" "$@"
echo "results: $out"
