# cuDL: A custom deep learning library in CUDA

cuDL is a deep learning library built from scratch for NVIDIA GPUs. It's written entirely in C++/CUDA, from the basic math up to training a neural network. It also includes testing, profiling and benchmarking tools to guide kernel optimization.

Each operation can have several kernel implementations, from simple to highly optimized. Every version is checked against a baseline CPU result and timed automatically, so performance can be compared without trading off accuracy.

Notes on adding kernels, benchmarking, debugging and profiling are in the [dev manual](docs/dev-manual.md).

## Project layout

```
include/cudl/   public headers
src/            library code (GPU code in src/kernels/)
tests/          correctness tests
bench/          speed benchmarks
scripts/        profiling helpers
docs/           dev manual
```

## Getting started

Prerequisites:
- An NVIDIA GPU with CUDA 12 or newer
- CMake 3.24+

Build:

```sh
cmake -S . -B build
cmake --build build -j
```

Run the tests (requires a GPU):

```sh
ctest --test-dir build --output-on-failure
```

Benchmark:

```sh
build/bench/bench_copy
build/bench/bench_elementwise
build/bench/bench_reduce
```

To build on a machine without a GPU (e.g. a cluster login node), see [Building without a GPU](docs/dev-manual.md#building-without-a-gpu). 
