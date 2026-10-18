# Common setup for the sbatch scripts in this directory; sourced, not run.
# Sends the job's output to logs/ next to this file and moves to the repo root, so jobs can be
# submitted from any directory: sbatch path/to/scripts/slurm/test.sbatch

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$here/logs"
exec >"$here/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>&1
# Slurm's default output file now only holds errors from before the redirect; drop it if there were none.
rm -f "${SLURM_SUBMIT_DIR}/slurm-${SLURM_JOB_ID}.out"

type module &>/dev/null || source /etc/profile.d/modules.sh
module load gcc/9.2.0 cmake/3.24.3 cuda12.3/toolkit/12.3.2 cuda12.3/nsight/12.3.2

cd "$(git -C "$here" rev-parse --show-toplevel)"
mkdir -p results profiles

if [[ ! -d build ]]; then
    echo "build/ not found: build on the login node first (see docs/dev-manual.md#running-on-slurm)" >&2
    exit 1
fi

echo "job=${SLURM_JOB_ID} node=$(hostname) commit=$(git rev-parse --short HEAD)$(git diff --quiet || echo -dirty)"
nvidia-smi --query-gpu=index,name,driver_version,clocks.max.sm,clocks.max.mem --format=csv
