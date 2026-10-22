#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace cudl::kernels {

// Optimizer update rules
// In-place application to one parameter tensor of n elements

// SGD with momentum (similar to PyTorch)
// momentum == 0 is plain SGD
// Variants:
//   scalar: one element per thread
using SgdFn = void (*)(float* param, const float* grad, float* velocity, size_t n, float lr, float momentum, cudaStream_t stream);

// Adam hyperparameters for one step 
// Caller computes the bias corrections for step t (1-indexed)
struct AdamStep {
    float lr;
    float beta1;
    float beta2;
    float eps;
    float bias_correction1;
    float bias_correction2;
};

// Adam optimizer 
// Variants:
//   scalar: one element per thread
// TODO: 8bit
using AdamFn = void (*)(float* param, const float* grad, float* m, float* v, size_t n, AdamStep hp, cudaStream_t stream);

// Kernel registration
void register_optim_kernels();
}  // namespace cudl::kernels
