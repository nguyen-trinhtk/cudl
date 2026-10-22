# cuDL: A custom deep learning library in CUDA

cuDL is a deep learning library built from scratch in C++ and CUDA. It starts from math kernels and tensor representations, then gradually builds up into the components needed to train a neural network: forward/backward passes, losses, and optimizers.

Each kernel can have multiple implementations, from simple baselines to optimized versions. Much of the optimization work focuses on general matrix multiplication (GEMM), since it is a core building block of neural networks. The library also includes optimized elementwise operations, reductions, losses, and optimizers.

The repo includes automatic correctness checks (including numerical drift detection) and benchmark sweeps. Notes on adding kernels, benchmarking, debugging, and profiling are in the [dev manual](docs/dev-manual.md).

## Project layout

```text id="jydwbf"
cudl/
├── include/cudl/       public headers
│   ├── core/           tensors, dtypes, NVTX helpers
│   ├── kernels/        kernel interfaces and registry
│   ├── nn/             layers and losses
│   └── optim/          optimizers
├── src/                library implementation
│   ├── core/
│   ├── kernels/        GPU kernels
│   ├── nn/
│   └── optim/
├── examples/           end-to-end programs
├── tests/              correctness tests
├── bench/              benchmarks
├── scripts/            data, profiling, and plotting tools
│   └── slurm/          scripts for running jobs on a cluster
└── docs/               developer documentation
```

## Getting started

Requires an NVIDIA GPU with CUDA 12+ and CMake 3.24+.

```sh id="t9ebzf"
cmake -S . -B build
cmake --build build -j
```

Run the tests:

```sh id="0s6b3a"
ctest --test-dir build --output-on-failure
```

Run benchmarks:

```sh id="gdnx2q"
build/bench/bench_copy
build/bench/bench_elementwise
build/bench/bench_reduce
build/bench/bench_gemm
build/bench/bench_nn
```

## Training a model

`examples/mlp_mnist.cpp` trains an MLP (784 → 512 → 256 → 10, ReLU) on MNIST using only cuDL kernels, then reports test accuracy and time per step.

```sh id="kqj10i"
scripts/get_mnist.sh
build/examples/mlp_mnist
```

The example uses Adam for 5 epochs by default, with options such as `--optim sgd`, `--hidden 256`, and `--epochs 10`.

The API is layer-based, similar to PyTorch (but without autograd):

```cpp id="4a1q7v"
nn::Rng rng(0);
nn::Sequential model;
model.add<nn::Linear>(784, 256, rng);
model.add<nn::ReLU>();
model.add<nn::Linear>(256, 10, rng);

nn::CrossEntropyLoss loss_fn;
optim::Adam opt(model.parameters(), 1e-3f);

Tensor loss = loss_fn.forward(model.forward(x), labels);
model.backward(loss_fn.backward());
opt.step();
```

For building on a machine without a GPU, such as a cluster login node, see [Building without a GPU](docs/dev-manual.md#building-without-a-gpu).