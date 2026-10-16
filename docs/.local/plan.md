# Roadmap

| Phase | Focus | Status |
|---|---|---|
| 0 | Groundwork: memory, testing and timing setup | ✅ Done |
| 1 | Basic math: element-wise operations, sums and maximums | Next |
| 2 | Matrix multiplication | Planned |
| 3 | Forward and backward passes for each operation | Planned |
| 4 | Neural network layers and training a small model end to end | Planned |
| 5 | Speedups from combining several steps into one GPU call | Planned |
| 6 | Optional: lower-precision math, convolutions and other speedups | Later |
| 7 | Optional: automatic optimization | Later |

Each phase ends with something that runs and a measurable exit check, so the library is usable after any phase.

## Phase 0: Skeleton and pipeline
**Goal:** the build, test and benchmark harness works before any real kernels exist.
- CMake project (CUDA language, arch flags, `CUDL_BUILD_TESTS`/`CUDL_BUILD_BENCH` options), GoogleTest, NVTX
- `core/`: `Tensor` (fp32, contiguous, strides kept), `Storage`, `Allocator` interface backed by `cudaMalloc`, `CUDA_CHECK`, `Context` (stream)
- Kernel registry keyed by `(op, variant)`
- Test utilities: random-tensor generator with seeds, `allclose` with an atol/rtol/ULP report
- Benchmark harness: warmup, cudaEvent timing, median/p10/p90, JSON-lines output with git SHA and GPU name
- One trivial kernel (`fill` or `copy`) running end to end through the registry, tests and benchmarks
- `scripts/profile.sh` (nsys and ncu wrappers)

**Exit:** `ctest` passes and `bench_copy` reports GB/s at more than 80% of peak memory bandwidth.

## Phase 1: Elementwise and reductions
**Goal:** the templated functor pattern and the first memory-bound kernels.
- Generic `elementwise_kernel<Op>` with unary and binary functors (add, mul, scale, relu, exp, …), plus simple broadcasting (bias over rows)
- Reductions: sum and max over all elements and over the last dimension. Variants: naive atomic → shared-memory tree → warp shuffle
- CPU references that accumulate in double
- First **drift test**: reduction error against N for each variant
- Determinism test, with atomic variants flagged as non-deterministic
- `bench/compare.py` to diff two benchmark runs

**Exit:** elementwise kernels reach about 80% of bandwidth, the best reduction reaches about 70%, the drift plot shows naive versus tree summation error growth, and all variants pass.

## Phase 2: GEMM
**Goal:** a contraction kernel family and a performance-tuning workflow.
- Variants: naive → coalesced → shared-memory tiled → register-tiled (2D blocking) → vectorized loads
- **Epilogue hook** `gemm<Epilogue>` from the start, with identity as the default
- Transpose flags (`A^T`, `B^T`), which the backward pass needs
- cuBLAS baseline in the benchmarks, sweeping square and skinny shapes
- Drift sweep over K (64 → 64k)
- An ncu profile write-up for each variant: occupancy, memory throughput, bank conflicts

**Exit:** the best variant reaches at least 50% of cuBLAS on 4096² fp32, and every variant passes the tests on odd and prime shapes.

## Phase 3: Ops with forward and backward passes
**Goal:** math semantics are separated from kernels, and gradients are verified.
- `ops/`: `add`, `mul`, `matmul`, `linear`, `relu`, `softmax`, `log_softmax`, `cross_entropy`, `mse`, each with `forward`, `backward`, `flops()`, `bytes()` and an op-kind tag
- Saved-context struct for backward
- **Gradcheck** framework: fp64 central finite differences on the CPU, compared with the GPU `backward`
- Numerically stable softmax (max-subtraction) with overflow edge tests

**Exit:** every op passes gradcheck and the edge-case suite.

## Phase 4: NN layer and training
**Goal:** an end-to-end model trains.
- `nn/`: `Module` (`forward`, `backward`, `parameters`), `Linear`, `ReLU`, `Sequential`, `CrossEntropyLoss`
- `optim/`: `SGD` (with momentum) and `Adam`
- Weight init (Kaiming/Xavier) and a seeded RNG
- `examples/mlp_mnist.cu` with a minimal binary MNIST loader
- **Long-run drift test:** N steps of GPU against the CPU reference with the same seed, tracking per-step parameter divergence
- End-to-end benchmark: step time broken down per op using NVTX in nsys

**Exit:** the MLP reaches at least 97% MNIST test accuracy, the drift stays bounded, and the per-op time breakdown is visible.

## Phase 5: Manual fusion (level 1 + 2)
**Goal:** measurable speedups from fusion, and a reusable fusion pattern.
- GEMM epilogues: bias, bias+ReLU, residual add
- Fused `softmax_cross_entropy` (forward and backward)
- Composed elementwise functors (`Compose<A,B,…>`)
- Fused optimizer step (multi-tensor Adam)
- Caching allocator (replaces raw `cudaMalloc`)
- Tests: fused against unfused composition with a fusion tolerance policy
- Benchmarks: fused versus unfused, per op and per training step, plus bytes saved

**Exit:** the MNIST step time drops by a measured percentage, the fused paths pass the tests, and the drift is no worse than unfused.

## Phase 6 (optional): Performance features
Pick whichever interests you:
- **Mixed precision:** fp16/bf16 storage with fp32 accumulation, loss scaling, and a dtype-keyed tolerance table
- **Tensor cores:** WMMA/MMA GEMM variant
- **CNN:** `conv2d` via im2col + GEMM, then a direct/implicit-GEMM variant
- **LayerNorm** (fused), with dropout and fused RNG
- **CUDA Graphs** to capture a training step

## Phase 7 (optional): Graph and automatic fusion (level 3)
- Lazy tensors and a graph IR, with `realize()`
- Tape-based autograd built on the graph (replaces layer-wise backward)
- Passes: dead-code elimination, elementwise-chain fusion, folding into GEMM epilogues
- NVRTC codegen for arbitrary elementwise chains, cached by graph hash
- Memory planning pass (buffer reuse)

**Exit:** auto-fused MNIST matches or beats the hand-fused Phase 5 numbers.

## Rules for every phase
- A kernel gets a CPU reference, registry entry, test and benchmark **in the same commit** as the kernel itself.
- Keep a `BENCHMARKS.md` log (or JSON history) to track performance across phases.
- After each phase, write a short note: what got faster, why (from ncu), and what drifted.

Phases 0–2 hold most of the learning; 3–5 are mostly plumbing on top of them.