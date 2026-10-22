#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace cudl::kernels {

// Every op is registered under two variants:
//   scalar: one element per thread
//   vec4: 4 floats per thread(falls back to scalar if misaligned)

// Unary ops: "neg", "relu", "exp", "tanh", "sigmoid".
using UnaryFn = void (*)(const float* in, float* out, size_t n, cudaStream_t stream);

// Scale (out = alpha * in)
using ScaleFn = void (*)(const float* in, float alpha, float* out, size_t n, cudaStream_t stream);

// Binary ops: "add", "sub", "mul", "div", "relu_grad"
// relu_grad(dy, x) = x > 0 ? dy : 0 (ReLU backward)
// Inputs must be same shape
using BinaryFn = void (*)(const float* a, const float* b, float* out, size_t n, cudaStream_t stream);

// Broadcast a row vector over a row-major matrix (so every row += bias once)
// Bias ops: "add_bias"
using BiasFn = void (*)(const float* x, const float* bias, float* out, size_t rows, size_t cols, cudaStream_t stream);

// Registration
inline constexpr const char* kUnaryOps[] = {"neg", "relu", "exp", "tanh", "sigmoid"};
inline constexpr const char* kBinaryOps[] = {"add", "sub", "mul", "div", "relu_grad"};

void register_elementwise_kernels();
}  // namespace cudl::kernels
