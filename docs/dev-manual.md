# cuDL dev manual

Notes on extending, debugging and profiling cuDL. Building and running are in the [README](../README.md).

## Adding a kernel

Kernels take raw device pointers and sizes, not Tensors. Every variant of an op shares one launcher signature, so tests and benchmarks can loop over all of them.

1. **Declare** the launcher signature and each variant in `include/cudl/kernels/<op>.h`:
   ```cpp
   using XxxFn = void (*)(const float* in, float* out, size_t n, cudaStream_t stream);
   void xxx_naive(const float* in, float* out, size_t n, cudaStream_t stream);
   void register_xxx_kernels();
   ```
2. **Implement** the kernels and launchers in `src/kernels/<op>.cu`. Call `CUDL_KERNEL_CHECK()` after every launch, and use `grid_size()` from `src/kernels/launch_utils.h` for grid-stride kernels.
3. **Register** the variants in `register_xxx_kernels()` using `detail::raw_registry<XxxFn>()`. Then call that function from `ensure_kernels_registered()` in `src/kernels/register_all.cpp`. Set `deterministic = false` for variants whose results depend on atomic ordering.
4. **Add the source file** to the `cudl` library in the top-level `CMakeLists.txt`.
5. **Write a CPU reference** in `tests/ref/<op>_ref.h`. It should be naive, obviously correct, and accumulate in `double`.
6. **Write a parameterized test** in `tests/test_<op>.cpp` over `registry<XxxFn>().variants("xxx")` × sizes × seeds, compared with `test::AllClose` (see `tests/test_copy.cpp`). Add the file to `tests/CMakeLists.txt`.
7. **Write a benchmark** in `bench/bench_<op>.cpp` that loops over `registry<XxxFn>().entries("xxx")` and passes the op's `bytes` and `flops` to the harness. Add it to `bench/CMakeLists.txt`.

Every registered variant is then tested and benchmarked automatically.

## Choosing a variant at run time

`registry<Fn>().get_default(op)` returns the first registered variant. Override it with an environment variable:

```sh
CUDL_KERNEL_COPY=naive build/bench/bench_copy
```

## Building without a GPU

On a machine without a GPU (e.g. a cluster login node), CMake can't detect the GPU type. Set it explicitly, for example `80` for an A100:

```sh
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=80
```

## Debugging

- **Async kernel faults:** configure with `-DCUDL_SYNC_KERNELS=ON`. Every launch is then followed by a device synchronize, so an error is reported at the kernel that caused it rather than at a later CUDA call.
- **Memory errors:** run a test binary under `compute-sanitizer build/tests/cudl_tests`.
- **One test:** `build/tests/cudl_tests --gtest_filter='*vec4*'`.

## Profiling

Reports are written to `profiles/` (override with `CUDL_PROFILE_DIR`).

```sh
# Timeline of all kernels and NVTX ranges
scripts/profile.sh nsys build/bench/bench_copy --iters 20

# Full metrics for one kernel (name is a regex)
scripts/profile.sh ncu copy_vec4 build/bench/bench_copy --warmup 0 --iters 1
```

Use `NCU_LAUNCH_SKIP` and `NCU_LAUNCH_COUNT` to choose which launches ncu profiles. Wrap new code in `CUDL_NVTX_RANGE("name")` so it shows up as a named block in nsys.

## Benchmarking

Each op has its own benchmark binary that times every registered variant across a range of sizes:

```sh
build/bench/bench_copy                        # results table to stdout
build/bench/bench_copy --json results.jsonl   # also append one JSON line per result
build/bench/bench_copy --flush-l2 --iters 200 # cold-cache numbers, more samples
```

Columns:
- **median_ms, p10_ms, p90_ms**: time per launch. Use the median; a wide p10–p90 spread means noisy measurements.
- **GB/s, GFLOP/s**: achieved throughput, from the op's `bytes` and `flops` cost model.
- **%peakBW**: GB/s as a percentage of the GPU's theoretical memory bandwidth.

For memory-bound ops, compare against the `cudaMemcpy` row: it's the practical ceiling, usually around 85–95% of the theoretical peak.

JSON results carry the GPU name, git commit and timestamp, so runs can be compared over time.

### Options

| Flag / variable | Effect |
|---|---|
| `--warmup N`, `--iters N` | Warmup and timed iteration counts (default 10 / 100) |
| `--flush-l2` | Clear L2 before every timed iteration (cold-cache numbers) |
| `--json PATH` or `CUDL_BENCH_JSON` | Append one JSON line per result |
| `CUDL_PEAK_BW_GBPS` | Override the theoretical peak bandwidth used for `%peakBW` |
| `CUDL_GIT_SHA` | Override the commit recorded in JSON output (default: captured at configure time) |

## Code style

4-space indentation, 120 columns, Google-based style, set in `.clang-format` and `.editorconfig`. To reformat everything:

```sh
clang-format -i $(git ls-files '*.h' '*.cpp' '*.cu')
```
