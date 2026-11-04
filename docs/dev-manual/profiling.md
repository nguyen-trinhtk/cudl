# Profiling

This doc explains how to profile a single kernel with ncu and a full training run with nsys, and what to look for in each report.

- [Profiling a kernel (ncu)](#profiling-a-kernel-ncu)
  - [Usage](#usage)
  - [Picking the launch](#picking-the-launch)
  - [Kernel names](#kernel-names)
  - [Reading the report](#reading-the-report)
- [Profiling training (nsys)](#profiling-training-nsys)
  - [Usage](#usage-1)
  - [Reading the results](#reading-the-results)
  - [Zooming in on one layer](#zooming-in-on-one-layer)
  - [Swapping variants](#swapping-variants)

Both go through `scripts/profile.sh`, which writes reports to `profiles/` (or `$CUDL_PROFILE_DIR` if set). `.ncu-rep` files open in Nsight Compute and `.nsys-rep` files open in Nsight Systems.

The [benchmarks](benchmarking.md) show which variant is slow, and profiling shows why, so it makes sense to benchmark first.

## Profiling a kernel (ncu)

### Usage

```text
[NCU_LAUNCH_SKIP=N] [NCU_LAUNCH_COUNT=N] scripts/profile.sh ncu KERNEL_REGEX COMMAND [ARGS...]
```

The usual target is a bench binary, filtered down to the one variant under investigation:

```sh
NCU_LAUNCH_SKIP=4 scripts/profile.sh ncu reduce_all_kernel \
    build/bench/bench_reduce --op sum --algo warp --load vec4 --warmup 0 --iters 1
```

**Note:** Always pass `--warmup 0 --iters 1`. ncu replays every launch it profiles dozens of times, so a normal bench run (110 launches per size) takes a very long time.

### Picking the launch

ncu profiles the launches whose name matches `KERNEL_REGEX`. Narrowing it down to one launch takes two steps:

- Variant: use the bench filters (`--op`, `--algo`, `--load`, see [picking variants](benchmarking.md#picking-variants)). Then only that variant launches, and the regex can just be the kernel name.
- Size: the bench runs sizes smallest first, one launch each. `NCU_LAUNCH_SKIP=N` skips the first N matching launches. `NCU_LAUNCH_COUNT=N` (default 1) profiles N in a row.

The `smem` and `warp` reductions launch twice per size (a partials pass, then a 1-block pass to combine them). That's why the example skips 4 to land on the partials pass of the 3rd size.

When the right skip count is unclear, setting `NCU_LAUNCH_COUNT=10` and picking the launch in the GUI also works. The grid size shows which data size each launch was.

### Kernel names

| Op | Kernel |
|---|---|
| `copy` | `copy_kernel` |
| unary, binary, `scale` | `elementwise_kernel` |
| `add_bias` | `bias_kernel` |
| `sum` `max` | `reduce_all_kernel` |
| `*_rows` | `reduce_rows_{thread,smem,warp}_kernel` |
| `*_cols` | `reduce_cols_thread_kernel` |
| `gemm` | `naive_kernel`, `smem_blocking` |
| `cross_entropy_loss` | `cross_entropy_loss_warp_kernel` |
| `sgd` `adam` | `sgd_kernel`, `adam_kernel` |

If this table is out of date, `grep -rn __global__ src/kernels` gives the real list.

**Note:** The regex matches the full demangled name, template args included, and enum args show up as numbers: `Load::Vec4` is `(cudl::kernels::Load)1`. That's why it's easier to filter with the bench flags than with the regex.

### Reading the report

Go through these in order:

- Speed of Light: memory % vs compute %. A memory-bound kernel should be up near what `cudaMemcpy` gets. If both are low, the kernel is latency-bound: not enough warps, or they're stalling.
- Memory Workload Analysis: if bytes transferred is way more than bytes requested, loads aren't coalesced. That's most of what `vec4` fixes.
- Occupancy and Warp State Stats: why it's latency-bound. `long_scoreboard` is waiting on memory, `barrier` is waiting on `__syncthreads`.
- Source (SASS): which line the stalls come from.

For GEMM, the roofline chart is worth checking too.

## Profiling training (nsys)

### Usage

```text
scripts/profile.sh nsys COMMAND [ARGS...]
```

```sh
scripts/profile.sh nsys build/examples/mlp_mnist --epochs 1
```

**Note:** One epoch is plenty. Every step looks the same, and the report gets huge with more epochs.

### Reading the results

Every layer, the loss and the optimizer are wrapped in NVTX ranges, so the timeline shows which kernels belong to which layer. The totals come from `nsys stats`:

```sh
nsys stats --report nvtx_gpu_proj_sum profiles/nsys-<stamp>.nsys-rep   # GPU time per range
nsys stats --report cuda_gpu_kern_sum profiles/nsys-<stamp>.nsys-rep   # GPU time per kernel
```

The report to use is `nvtx_gpu_proj_sum`, not `nvtx_sum`. Launches are async, so the CPU time of a range is mostly launch overhead, not the work.

If a range's GPU time is a lot smaller than its CPU time, or the timeline has gaps between kernels, the step is bound by launch overhead and faster kernels won't help.

The ranges are `train_step`, `load_batch`, `evaluate`, `linear.forward`, `linear.backward`, `relu.forward`, `relu.backward`, `cross_entropy.forward`, `sgd.step` and `adam.step`. A new range is added by putting `CUDL_NVTX_RANGE("name")` at the top of a scope.

### Zooming in on one layer

Once the slow layer is known, ncu can be limited to the kernels inside its range:

```sh
ncu --set full --nvtx --nvtx-include 'linear.backward/' --launch-count 3 \
    -o profiles/ncu-linear-backward build/examples/mlp_mnist --epochs 1
```

### Swapping variants

To check whether another variant would make training faster, override the one a layer uses and profile again:

```text
CUDL_KERNEL_<OP>=<VARIANT> COMMAND [ARGS...]
```

```sh
CUDL_KERNEL_GEMM=naive build/examples/mlp_mnist --epochs 1
CUDL_KERNEL_RELU=scalar build/examples/mlp_mnist --epochs 1
```

The variant name is the algo (`naive`), the algo plus `_vec4` (`warp_vec4`), or just `scalar`/`vec4` for ops with no algo.

**Note:** Don't set `CUDL_KERNEL_SUM`. The override hits every lookup of `sum`, and the column sum in `linear.backward` only has `thread`, so it crashes.
