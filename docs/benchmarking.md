# Benchmarking

This doc explains how to run the kernel benchmarks, pick which variants to run, and read and plot the results.

- [Running benchmarks](#running-benchmarks)
  - [Usage](#usage)
  - [Flags](#flags)
  - [Picking variants](#picking-variants)
  - [Reading the output](#reading-the-output)
- [Plotting results](#plotting-results)
  - [Usage](#usage-1)
  - [Reading the chart](#reading-the-chart)

## Running benchmarks

### Usage

```text
build/bench/bench_<family> [--warmup N] [--iters N] [--flush-l2] [--json FILE]
                           [--op REGEX] [--algo REGEX] [--load REGEX]
```

There is one binary per family of ops. Each one runs every variant of its ops over a few data sizes and prints a table.

```sh
build/bench/bench_copy
build/bench/bench_elementwise
build/bench/bench_reduce
build/bench/bench_gemm
build/bench/bench_nn
```

### Flags

```sh
--warmup 10 --iters 100                   # untimed + timed runs (defaults)
--flush-l2                                # cold cache before each timed run
--json out.jsonl                          # append results to a file
--op sum --algo 'smem|warp' --load vec4   # variant filters
```

**Note:** The defaults are fine for comparing variants. If p10 and p90 are far apart, the run is noisy and `--iters` should be raised. `--flush-l2` is worth using for kernels that only read their data once in real use, since otherwise the small sizes sit in L2 and look faster than they are.

### Picking variants

A kernel variant is defined by three things:

- Op: what it computes (`sum`, `relu`, `gemm`, ...)
- Algo: how it's implemented on the GPU (`naive`, `smem`, `warp`, ...)
- Load: how it reads memory, one float at a time (`scalar`) or four (`vec4`)

Each one has a filter flag (`--op`, `--algo`, `--load`) that takes a regex. Available values:

| Binary | `--op` | `--algo` | `--load` |
|---|---|---|---|
| `bench_copy` | `copy` | none | `scalar` `vec4` |
| `bench_elementwise` | `neg` `relu` `exp` `tanh` `sigmoid` `scale` `add` `sub` `mul` `div` `relu_grad` `add_bias` | none | `scalar` `vec4` |
| `bench_reduce` | `sum` `max` | `atomic` `smem` `warp` | `scalar` `vec4` |
| | `sum_rows` `max_rows` | `thread` `smem` `warp` | `scalar` |
| | `sum_cols` `max_cols` | `thread` | `scalar` |
| `bench_gemm` | `gemm` | `naive` `smem` `smem_1d` `smem_2d` | `scalar` `vec4` (`smem_2d` only) |
| `bench_gemm_tune` | `gemm` | `smem_2d_<BM>x<BN>x<BK>_<TM>x<TN>` | `vec4` |
| `bench_nn` | `cross_entropy_loss` | `warp` | `scalar` |
| | `sgd` `adam` | none | `scalar` |

Filter rules:

- The regex has to match the whole name. `--op sum` won't match `sum_rows`, use `--op 'sum.*'` for both.
- Use `|` to pick several: `--op 'add|mul'`. Quote it so the shell doesn't eat it.
- Filters are AND-ed. Leave one out to match everything on that axis.
- Ops with no algo (only one implementation) get skipped by any `--algo` filter.

### Tuning GEMM tile sizes

`bench_gemm_tune` times the `smem_2d` vec4 kernel for every combination of BM, BN = 64, 128, 256, BK = 8, 16, 32 and TM, TN = 4, 8. Configs the kernel can't run, or with fewer than 64 threads, are skipped. Each config is checked against `naive` before it is timed, and a ranking by GFLOP/s is printed at the end. The default size is 4096.

```sh
build/bench/bench_gemm_tune --size 4096 --json tune.jsonl
build/bench/bench_gemm_tune --algo 'smem_2d_128x128x.*'    # only BM = BN = 128
```

To try other sizes, edit `kBM`...`kTN` in `bench/bench_gemm_tune.cu` and rebuild.

### Reading the output

Variants are ranked by `median_ms` first. After that, the metric to check depends on the kernel:

- GEMM (compute-bound): `GFLOP/s`.
- Everything else (memory-bound): `%peakBW`. `cudaMemcpy` gets about 85-95%, which is about as good as it gets, so a kernel close to that has little left to gain.

For GEMM, the roofline chart in `ncu` is also worth a look (see [profiling](profiling.md)).

## Plotting results

### Usage

```text
scripts/plot_bench.py FILE [-o PNG] [--metric gbps|pct|ms] [--op REGEX] [--algo REGEX] [--load REGEX] [--cols N]
```

A run has to be saved to a file with `--json` before it can be plotted:

```sh
build/bench/bench_reduce --json run.jsonl
scripts/plot_bench.py run.jsonl -o reduce.png
```

Options:

```sh
--metric gbps|pct|ms         # y axis: GB/s (default), % of peak bandwidth, or time
--op sum --algo 'smem|warp'  # same filters as the bench
--cols 3                     # charts per row
```

**Note:** `--json` appends instead of overwriting. If a file has the same kernel twice, the plot uses the newest one.

### Reading the chart

The script draws one chart per op. Each group of bars is one data size, and each bar is one variant:

- Bar color is the algo. Solid bars are `scalar` loads, striped bars are `vec4`.
- Taller is better (for the default `gbps`).
- The dashed line is `cudaMemcpy`, the practical ceiling. It only shows up for `bench_copy` and `bench_elementwise`, since they're the only ones with a memcpy row.
