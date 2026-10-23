# Testing

This doc explains how to run the tests, what each suite checks, and how to run the extra drift and memory-error checks.

- [Running tests](#running-tests)
  - [Usage](#usage)
  - [Picking tests](#picking-tests)
  - [What's tested](#whats-tested)
- [Drift plots](#drift-plots)
- [Memory errors](#memory-errors)

## Running tests

### Usage

```text
ctest --test-dir build [--output-on-failure] [-R REGEX]
build/tests/cudl_tests [--gtest_filter=PATTERN]
```

Everything is GoogleTest, built into one binary, `build/tests/cudl_tests`. `ctest` runs that binary one test at a time.

```sh
ctest --test-dir build --output-on-failure
```

**Note:** Every registered kernel variant gets tested automatically, so a new variant is covered as soon as it's in the registry. New tests are only needed for new ops.

### Picking tests

```sh
ctest --test-dir build -R Reduce                  # regex, matches anywhere in the name
build/tests/cudl_tests --gtest_filter='*Gemm*'    # glob, has to match the whole name
```

Parameterized tests are named like `AllVariants/ReduceAllTest.<case>/<variant>`, so put `*` on both sides in `--gtest_filter`.

Running the binary directly is faster when iterating on one thing, and it prints everything, not just failures.

### What's tested

| Suite | Checks |
|---|---|
| `*KernelTest`, `*EdgeTest` | each variant vs a CPU reference in `tests/ref/`, incl. odd sizes and misaligned pointers |
| `ReduceDeterminism` | same input gives bit-identical results every run (`sum`/`atomic` uses float atomics, so it's only reported) |
| `ReduceDrift` | sum error vs N per variant; tree reductions (`smem`, `warp`) must stay under 1e-5 |
| `LinearTest`, `Gradcheck` | layer forward/backward vs CPU, gradients vs finite differences |
| `Training` | Adam can overfit one batch |
| `TrainingDrift` | GPU training loss stays close to an fp64 CPU run |

Test buffers are padded with sentinels (`tests/common/guarded.h`), so writing out of bounds fails the test.

## Drift plots

```text
CUDL_DRIFT_CSV=FILE build/tests/cudl_tests --gtest_filter='ReduceDrift.*'
CUDL_TRAIN_DRIFT_CSV=FILE build/tests/cudl_tests --gtest_filter='TrainingDrift.*'
scripts/plot_drift.py FILE [-o PNG]
```

Float sums lose precision as N grows, and how fast depends on the algorithm. The drift tests measure that error, and these env vars dump it to a CSV so the trend is visible instead of just pass/fail:

```sh
CUDL_DRIFT_CSV=drift.csv build/tests/cudl_tests --gtest_filter='ReduceDrift.*'
scripts/plot_drift.py drift.csv -o drift.png
```

The plot is error vs N (log-log), one line per sum variant plus two CPU baselines. `cpu_sequential` adds one value at a time, so its error climbs with N. Tree reductions (`smem`, `warp`) should stay down near `cpu_pairwise`.

## Memory errors

```text
compute-sanitizer --error-exitcode 1 build/tests/cudl_tests [--gtest_filter=PATTERN]
```

compute-sanitizer catches out-of-bounds and misaligned accesses that the sentinels miss, such as bad reads.

**Note:** It is slow, so it's best to filter down to the tests for whatever changed. On the cluster, `CUDL_SANITIZE=1 sbatch scripts/slurm/test.sbatch` runs the full thing after ctest.
