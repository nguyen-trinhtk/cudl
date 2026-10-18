#!/bin/bash
# Run the test suite on one GPU. Extra arguments go to ctest.
#
#   sbatch scripts/slurm/test.sbatch
#   sbatch scripts/slurm/test.sbatch -R Reduce
#   CUDL_SANITIZE=1 sbatch scripts/slurm/test.sbatch   # also run under compute-sanitizer
#SBATCH -J cudl-test
#SBATCH -A spiralfreeprj
#SBATCH --qos=free
#SBATCH -p gpu-sm
#SBATCH --gres=gpu:v100:1
#SBATCH -c 4
#SBATCH -t 00:30:00
set -euo pipefail
# Slurm runs a copy of this script, so find the original (as submitted) to locate env.sh.
source "$(dirname "$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/^ *Command=\([^ ]*\).*/\1/p')")/env.sh"

ctest --test-dir build --output-on-failure "$@"

if [[ -n "${CUDL_SANITIZE:-}" ]]; then
    compute-sanitizer --error-exitcode 1 build/tests/cudl_tests
fi
