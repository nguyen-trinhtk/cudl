#!/bin/bash
# Train the MNIST MLP on one GPU (Phase 4 exit check). Extra arguments go to mlp_mnist.
# Download the data on the login node first: scripts/get_mnist.sh
#
#   sbatch scripts/slurm/train.sbatch
#   sbatch scripts/slurm/train.sbatch --optim sgd --epochs 10
#SBATCH -J cudl-train
#SBATCH -A spiralfreeprj
#SBATCH --qos=free
#SBATCH -p gpu-sm
#SBATCH --gres=gpu:v100:1
#SBATCH -c 4
#SBATCH -t 00:30:00
set -euo pipefail
# Slurm runs a copy of this script, so find the original (as submitted) to locate env.sh.
source "$(dirname "$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/^ *Command=\([^ ]*\).*/\1/p')")/env.sh"

build/examples/mlp_mnist "$@"
