# Running on Slurm

This doc explains how to build cuDL on the cluster and run tests, benchmarks, profiles and training as Slurm jobs.

- [Building](#building)
- [Submitting jobs](#submitting-jobs)
  - [Usage](#usage)
  - [Scripts](#scripts)
  - [Logs and output](#logs-and-output)
- [Interactive shell](#interactive-shell)

## Building

```sh
module load gcc/9.2.0 cmake/3.24.3 cuda12.3/toolkit/12.3.2
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=70    # 70 = V100
cmake --build build -j
```

**Note:** Build on the login node, not in a job. Compute nodes might not have internet, and cmake needs it to pull GoogleTest. Building doesn't need a GPU.

## Submitting jobs

### Usage

```text
[ENV=VALUE] sbatch [SBATCH_OPTIONS] scripts/slurm/<script>.sbatch [ARGS...]
```

Anything after the script name gets passed through to whatever the script runs. sbatch options (like `-t` or `-p`) go before the script.

```sh
sbatch scripts/slurm/bench.sbatch --op sum
sbatch -t 02:00:00 scripts/slurm/bench.sbatch
```

Each job asks for one V100 on `gpu-sm` and can be submitted from any directory.

**Note:** A job runs whatever is in `build/` when it starts, not when it was submitted. Rebuild before submitting, and avoid rebuilding while a job is still queued.

### Scripts

| Script | Runs | Args go to | Time limit |
|---|---|---|---|
| `test.sbatch` | `ctest` | `ctest` | 30 min |
| `bench.sbatch` | every bench binary except `bench_gemm_tune` | each bench binary | 1 h |
| `tune_gemm.sbatch` | `bench_gemm_tune` (GEMM tile sweep) | `bench_gemm_tune` | 30 min |
| `profile.sbatch` | `scripts/profile.sh` | `profile.sh` | 30 min |
| `train.sbatch` | `build/examples/mlp_mnist` | `mlp_mnist` | 30 min |

```sh
sbatch scripts/slurm/test.sbatch -R Reduce
CUDL_SANITIZE=1 sbatch scripts/slurm/test.sbatch          # also runs compute-sanitizer
sbatch scripts/slurm/bench.sbatch --load vec4
sbatch scripts/slurm/profile.sbatch nsys build/examples/mlp_mnist --epochs 1
sbatch scripts/slurm/train.sbatch --optim sgd
```

**Note:** For `train.sbatch`, download MNIST on the login node first with `scripts/get_mnist.sh`, for the same no-internet reason.

### Logs and output

- Job log: `scripts/slurm/logs/<job-name>-<jobid>.out`. It starts with the node, commit and GPU, which shows exactly what ran.
- If the job died before that, the error is in `slurm-<jobid>.out` in the directory the job was submitted from.
- Bench results: `results/bench-<commit>-<jobid>.jsonl` (`results/tune-gemm-...` for the sweep), ready for [plotting](benchmarking.md#plotting-results).
- Profiles: `profiles/`.

## Interactive shell

```sh
srun -A spiralfreeprj --qos=free -p gpu-sm --gres=gpu:1 --pty bash
```

An interactive shell is handy for poking around, running one test, or profiling something that needs several reruns. Anything long should go in a job instead, so it doesn't die when the session ends.
