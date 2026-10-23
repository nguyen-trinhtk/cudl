# cuDL

## Introduction 

This is a small deep learning library written from scratch in C++/CUDA, with a particular focus on general matrix multiplication (GEMM) implementation and optimization. 

It starts with basic math kernels (e.g., copy, elementwise operations, reductions, and GEMM) and gradually builds up to NN forward/backward passes, losses, and optimizers. Most kernels have a naive implementation alongside several progressively optimized versions.

Correctness tests (with numerical drift checks) and benchmarks are included throughout. The repo also demonstrates an example of how to build an MLP with the library and train it on MNIST.

## Relevant docs

- [Benchmarking](docs/benchmarking.md)
- [Profiling](docs/profiling.md)
- [Testing](docs/testing.md)
- [Running on Slurm](docs/slurm.md)
- [MLP on MNIST example](examples/README.md)

## Repo layout

```text
cudl/
├── include/cudl/       public headers
│   ├── core/           tensors, dtypes, nvtx helpers
│   ├── kernels/        kernel interfaces + registry
│   ├── nn/             layers, losses
│   └── optim/          optimizers
├── src/                implementation
│   ├── core/           tensors, allocator
│   ├── kernels/        CUDA kernels + launchers
│   │   └── gemm/       one header per GEMM kernel
│   ├── nn/             layers, losses
│   └── optim/          optimizers
├── examples/
├── tests/
├── bench/
├── scripts/            data, profiling, plotting
│   └── slurm/
└── docs/
```

## Getting started

### Build

**Requirements:** NVIDIA GPU with CUDA 12+ and CMake 3.24+

```sh
cmake -S . -B build
cmake --build build -j
```

### Test

```sh
ctest --test-dir build --output-on-failure
```

### Benchmark

```sh
build/bench/bench_copy
build/bench/bench_elementwise
build/bench/bench_reduce
build/bench/bench_gemm
build/bench/bench_nn
```
