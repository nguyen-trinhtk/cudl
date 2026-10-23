#!/bin/bash
# Run scripts/profile.sh on one GPU; takes the same arguments. Reports go to profiles/.
#
#   sbatch scripts/slurm/profile.sbatch nsys build/bench/bench_copy --iters 20
#   sbatch scripts/slurm/profile.sbatch ncu copy_kernel build/bench/bench_copy --load vec4 --warmup 0 --iters 1
#SBATCH -J cudl-profile
#SBATCH -A spiralfreeprj
#SBATCH --qos=free
#SBATCH -p gpu-sm
#SBATCH --gres=gpu:v100:1
#SBATCH -c 4
#SBATCH -t 00:30:00
set -euo pipefail
# Slurm runs a copy of this script, so find the original (as submitted) to locate env.sh.
source "$(dirname "$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/^ *Command=\([^ ]*\).*/\1/p')")/env.sh"

scripts/profile.sh "$@"
