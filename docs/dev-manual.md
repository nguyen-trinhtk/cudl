# cuDL dev manual

Notes on extending, debugging and profiling cuDL. Building and running are in the [README](../README.md).

- [Adding a kernel](#adding-a-kernel)
  - [Kernel file layout](#kernel-file-layout)
  - [Variant axes](#variant-axes)
  - [Adding an elementwise op](#adding-an-elementwise-op)
  - [Reductions](#reductions)
- [Determinism and drift](#determinism-and-drift)
- [Choosing a variant at run time](#choosing-a-variant-at-run-time)
- [Building without a GPU](#building-without-a-gpu)
- [Running on Slurm](#running-on-slurm)
- [Debugging](#debugging)
- [Profiling](#profiling)
- [Benchmarking](#benchmarking)
  - [Options](#options)
- [Code style](#code-style)

## Adding a kernel

Kernels take raw device pointers and sizes, not Tensors. Every variant of an op shares one launcher signature, so tests and benchmarks can loop over all of them.

1. **Declare** the launcher signature and the registration function in `include/cudl/kernels/<op>.h`, with a comment listing the variants. The launchers themselves stay internal to the `.cu` file; callers go through the registry.
   ```cpp
   // Variants:
   //   scalar: ...
   //   vec4:   ...
   using XxxFn = void (*)(const float* in, float* out, size_t n, cudaStream_t stream);
   void register_xxx_kernels();
   ```
2. **Implement** the kernels and launchers in `src/kernels/<op>.cu`, following the [kernel file layout](#kernel-file-layout). Call `CUDL_KERNEL_CHECK()` after every launch, and use `grid_size()` from `src/kernels/launch_utils.h` for grid-stride kernels.
3. **Register** the variants in `register_xxx_kernels()` using `detail::raw_registry<XxxFn>()`, usually with `add_loads()` (see [variant axes](#variant-axes)). Then call that function from `ensure_kernels_registered()` in `src/kernels/register_all.cpp`. Set `deterministic = false` for variants whose results depend on atomic ordering.
4. **Add the source file** to the `cudl` library in the top-level `CMakeLists.txt`.
5. **Write a CPU reference** in `tests/ref/<op>_ref.h`. It should be naive, obviously correct, and accumulate in `double`.
6. **Write a parameterized test** in `tests/test_<op>.cpp` over `registry<XxxFn>().variants("xxx")` × sizes × seeds, compared with `test::AllClose` (see `tests/test_copy.cpp`). Add the file to `tests/CMakeLists.txt`.
7. **Write a benchmark** in `bench/bench_<op>.cpp` that loops over `registry<XxxFn>().entries("xxx")` and passes the op's `bytes` and `flops` to the harness. Add it to `bench/CMakeLists.txt`.

Every registered variant is then tested and benchmarked automatically.

### Kernel file layout

A kernel file is organized in layers, from device code up to registration. Each layer is marked with a `// ---- <Layer> ----` comment, in this order (see `src/kernels/reduce.cu` and `src/kernels/elementwise.cu`):

| Layer | Runs on | What goes here | Examples |
|---|---|---|---|
| **Functors** | device | Structs with `__device__ operator()`, plus `identity()` / `atomic()` when the op needs them. Shared ones go in their own `.cuh`. | `SumOp`, `MaxOp`, `elementwise_ops.cuh` |
| **Device helpers** | device | `__device__` building blocks the kernels reuse: per-element apply, per-thread partials, block/warp reductions. | `apply4`, `load_partial`, `block_reduce_warp` |
| **Kernels** | device | `__global__` kernels, named `<op>_kernel`, templated on the variant axes and then `Op`. Use `if constexpr` to choose a path, never a runtime flag, so each variant is its own kernel symbol (separate in `ncu`, no unused path in registers). | `reduce_all_kernel<Variant, Load, Op>` |
| **Host helpers** | host | `launch_<op>(Load l, ...)` turns the run-time choice into a template argument, launches, and calls `CUDL_KERNEL_CHECK()`. Grid-size helpers go here too. | `launch_reduce_all`, `partial_blocks` |
| **Launchers** | host | Functions with the registry signature (`XxxFn`), templated on the variant axes. They handle `n == 0`, check alignment, and plan the passes; they don't contain `<<<>>>` themselves unless there is only one instantiation. | `reduce_all<V, L, Op>`, `unary<L, Op>` |
| **Registration** | host | `add_<op>` helpers and `register_<op>_kernels()`, one `add` line per variant. | `add_reduce`, `add_unary` |

Shared pieces go in `src/kernels/launch_utils.h` rather than being redefined per file: `grid_size`, `aligned16`, `work_items`, `enum class Load { Scalar, Vec4 }`, and the hardware constants `K_WARP`, `K_MAX_BLOCK` and `MAX_NUM_WARP`.

Block size is a per-kernel tuning knob, so each file defines its own `constexpr int K_BLOCK = K_DEFAULT_BLOCK;` at the top and overrides it if that kernel tunes differently. Shared-memory arrays sized by the block (e.g. `smem[K_BLOCK]`) must use the file's `K_BLOCK`, and it must be a multiple of `K_WARP` if the file uses warp shuffles.

### Variant axes

Every registered kernel is identified by three independent axes, so variants can be compared one change at a time:

| Axis | Meaning | Values | In code |
|---|---|---|---|
| **op** | what is computed | `sum`, `max`, `relu`, `add_bias`, ... | registry key |
| **algo** | how it is computed | e.g. `atomic` / `smem` / `warp` for reductions; empty for ops with no algorithm choice (elementwise, copy) | op-specific `enum class Variant` template parameter |
| **load** | how each thread reads memory | `scalar` / `vec4` | `enum class Load` template parameter (`launch_utils.h`) |

A `vec4` kernel falls back to scalar loads at run time when a pointer isn't 16-byte aligned (or, for bias, when `cols % 4 != 0`).

`KernelEntry` stores `op`, `algo` and `load` as separate fields. `Registry::add` builds the unique `variant` name from the last two: `<algo>` for scalar loads, `<algo>_vec4` for vector loads, and just `scalar` / `vec4` when there is no algo. That name is what tests, `CUDL_KERNEL_<OP>` and `registry.get()` use.

Register both loads of one algo with `add_loads`:

```cpp
add_loads(ra, op, "smem", &reduce_all<Variant::Smem, Load::Scalar, Op>, &reduce_all<Variant::Smem, Load::Vec4, Op>);
add_loads(detail::raw_registry<UnaryFn>(), op, "", &unary<Load::Scalar, Op>, &unary<Load::Vec4, Op>);  // no algo axis
```

### Adding an elementwise op

Elementwise ops share one templated kernel, so a new op is a functor plus one registration line:

1. Add a struct with a `__device__ float operator()` to `src/kernels/elementwise_ops.cuh`. Unary functors take one float, binary take two; state such as a scalar goes in a member (see `Scale`).
2. Register it in `register_elementwise_kernels()` in `src/kernels/elementwise.cu` with `add_unary<ops::Xxx>("xxx")` or `add_binary<...>`. This registers both the `scalar` and `vec4` variants.
3. Add the name to `kUnaryOps` or `kBinaryOps` in `include/cudl/kernels/elementwise.h`, and a double-precision reference to `tests/ref/elementwise_ref.h`. The tests and `bench_elementwise` pick it up from the list.

### Reductions

Multi-block reductions write one partial result per block, then reduce the partials with a second single-block launch. The caller passes the scratch buffer for those partials (`kReduceWorkspaceFloats` floats, see `include/cudl/kernels/reduce.h`), so the kernels never allocate.

## Determinism and drift

- **Determinism:** `ReduceDeterminism.*` runs each reduction 10 times on the same input. Variants registered with `deterministic = true` must give bit-identical results; the others (atomic sum) only report how many runs differed.
- **Drift:** `ReduceDrift.SumErrorVsN` prints the relative error of every sum variant against the double reference for N = 2^10 … 2^24, next to fp32 sequential and pairwise CPU sums. To plot it (needs matplotlib):
  ```sh
  CUDL_DRIFT_CSV=drift.csv build/tests/cudl_tests --gtest_filter='ReduceDrift.*'
  scripts/plot_drift.py drift.csv -o drift.png
  ```

## Choosing a variant at run time

`registry<Fn>().get_default(op)` returns the first registered variant. Override it with an environment variable:

```sh
CUDL_KERNEL_SUM=warp_vec4 build/tests/cudl_tests   # registry.get_default("sum") now returns warp_vec4
```

## Building without a GPU

On a machine without a GPU (e.g. a cluster login node), CMake can't detect the GPU type. Set it explicitly, for example `80` for an A100:

```sh
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=80
```

## Running on Slurm

`scripts/slurm/` has batch scripts for the tests, benchmarks and profiler. Each one requests one V100 on `gpu-sm`. Build on the login node first (the configure step downloads GoogleTest, and compute nodes may be offline):

```sh
module load gcc/9.2.0 cmake/3.24.3 cuda12.3/toolkit/12.3.2
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build build -j
```

Then submit with `sbatch`, from any directory. The job runs from the root of the repo the script is in, and its output (stdout and stderr) goes to `scripts/slurm/logs/<job-name>-<jobid>.out`:

```sh
sbatch scripts/slurm/test.sbatch                    # extra args go to ctest
sbatch scripts/slurm/bench.sbatch --load vec4       # extra args go to every bench binary
sbatch scripts/slurm/profile.sbatch nsys build/bench/bench_copy --iters 20
sbatch -t 02:00:00 -p gpu scripts/slurm/bench.sbatch   # sbatch options go before the script
```

`#SBATCH` lines can't use variables and Slurm runs a copy of the job script, so each script finds the original through `scontrol show job` and `env.sh` redirects the output itself. If a job fails before that point, its error is in `slurm-<jobid>.out` in the directory you submitted from.

Benchmark JSON goes to `results/` and profiler reports to `profiles/`. A job runs whatever is in `build/` when it starts, so rebuild before submitting. For an interactive GPU shell, use `srun -A spiralfreeprj --qos=free -p gpu-sm --gres=gpu:1 --pty bash`.

## Debugging

- **Async kernel faults:** configure with `-DCUDL_SYNC_KERNELS=ON`. Every launch is then followed by a device synchronize, so an error is reported at the kernel that caused it rather than at a later CUDA call.
- **Memory errors:** run a test binary under `compute-sanitizer build/tests/cudl_tests`.
- **One test:** `build/tests/cudl_tests --gtest_filter='*vec4*'`.

## Profiling

Reports are written to `profiles/` (override with `CUDL_PROFILE_DIR`).

```sh
# Timeline of all kernels and NVTX ranges
scripts/profile.sh nsys build/bench/bench_copy --iters 20

# Full metrics for one kernel (regex over the demangled name, template args included)
scripts/profile.sh ncu 'copy_kernel<.*Vec4' build/bench/bench_copy --warmup 0 --iters 1
scripts/profile.sh ncu 'reduce_all_kernel<.*Warp.*Vec4' build/bench/bench_reduce --warmup 0 --iters 1
```

Use `NCU_LAUNCH_SKIP` and `NCU_LAUNCH_COUNT` to choose which launches ncu profiles. Wrap new code in `CUDL_NVTX_RANGE("name")` so it shows up as a named block in nsys.

## Benchmarking

Each op has its own benchmark binary that times every registered variant across a range of sizes:

```sh
build/bench/bench_copy                        # results table to stdout
build/bench/bench_copy --json results.jsonl   # also append one JSON line per result
build/bench/bench_copy --flush-l2 --iters 200 # cold-cache numbers, more samples
```

Select kernels along each [axis](#variant-axes) with `--op`, `--algo` and `--load`. Each takes a regex that must match the whole name, and unselected kernels are never launched (useful under `ncu`). Baseline rows such as `cudaMemcpy` always run.

```sh
build/bench/bench_reduce --load vec4                       # load axis: vec4 kernels only
build/bench/bench_reduce --algo 'smem|warp' --load scalar  # algo axis, load held fixed
build/bench/bench_reduce --op sum --algo warp              # one op, both loads of one algo
build/bench/bench_elementwise --op 'add|mul'               # op axis
```

Columns:
- **op, algo, load**: the kernel's [axes](#variant-axes). `algo` is `-` for ops without one; baseline rows show the baseline's name as `algo` and `-` as `load`.
- **median_ms, p10_ms, p90_ms**: time per launch. Use the median; a wide p10–p90 spread means noisy measurements.
- **GB/s, GFLOP/s**: achieved throughput, from the op's `bytes` and `flops` cost model.
- **%peakBW**: GB/s as a percentage of the GPU's theoretical memory bandwidth.

For memory-bound ops, compare against the `cudaMemcpy` row: it's the practical ceiling, usually around 85–95% of the theoretical peak.

JSON results carry the GPU name, git commit and timestamp, so runs can be compared over time. To diff two runs:

```sh
build/bench/bench_reduce --json before.jsonl
# ... change a kernel, rebuild ...
build/bench/bench_reduce --json after.jsonl
bench/compare.py before.jsonl after.jsonl --threshold 5
```

Rows are matched on GPU, op, algo, load and shape. `compare.py` takes the same `--op`, `--algo` and `--load` filters as the benchmarks, plus `--filter`, a regex searched in `op/variant/shape`. Changes beyond the threshold are marked `+` (faster) or `-` (slower), and the script exits with 1 if anything got slower.

### Options

| Flag / variable | Effect |
|---|---|
| `--warmup N`, `--iters N` | Warmup and timed iteration counts (default 10 / 100) |
| `--flush-l2` | Clear L2 before every timed iteration (cold-cache numbers) |
| `--json PATH` or `CUDL_BENCH_JSON` | Append one JSON line per result (includes `op`, `algo`, `load` and `variant`) |
| `--op`, `--algo`, `--load REGEX` | Run only kernels whose axis fully matches the regex |
| `CUDL_PEAK_BW_GBPS` | Override the theoretical peak bandwidth used for `%peakBW` |
| `CUDL_GIT_SHA` | Override the commit recorded in JSON output (default: captured at configure time) |

## Code style

4-space indentation, 120 columns, Google-based style, set in `.clang-format` and `.editorconfig`. To reformat everything:

```sh
clang-format -i $(git ls-files '*.h' '*.cpp' '*.cu')
```
