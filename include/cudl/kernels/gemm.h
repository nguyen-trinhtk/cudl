#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace cudl::kernels {

// BLAS's general matrix multiplication (GEMM)
// C = alpha AB + beta C
// where A = MxK, B = KxN, C = MxN
// Setting beta = 0 to ignore existing C-valyes

// Variants:
//  naive: each thread compute each Cij
using GemmFn = void (*)(const float* a, const float* b, float* c, size_t m, size_t n, size_t k, float alpha,
                       float beta, cudaStream_t stream);

// Kernel registration
void register_gemm_kernels();
}  // namespace cudl::kernels
