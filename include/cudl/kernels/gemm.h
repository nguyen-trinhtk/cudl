#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace cudl::kernels {

// BLAS's general matrix multiplication (GEMM)
// C = alpha op(A) op(B) + beta C
// where op(A) = MxK, op(B) = KxN, C = MxN, all row-major
// op(X) = X, or X^T when the trans flag is set:
//   trans_a: A is stored KxM     trans_b: B is stored NxK
// Setting beta = 0 to ignore existing C-values

// Variants:
//  naive: each thread compute each Cij
//  smem: 32x32 tiles of A and B staged in shared memory, each thread still computes one Cij
//  smem_1d: 64x64 C tile per block with 64x8 / 8x64 tiles of A and B, each thread computes 8 Cij in one column
//  smem_2d: same tiles as smem_1d with 8x8 threads per block, each thread computes an 8x8 tile of Cij, A tile stored transposed
//  smem_2d_vec4: smem_2d with float4 global loads/stores and smem reads
//                (falls back to scalar for transposed inputs, unaligned pointers, or k / n not multiple of 4)
using GemmFn = void (*)(const float* a, const float* b, float* c, size_t m, size_t n, size_t k, bool trans_a,
                       bool trans_b, float alpha, float beta, cudaStream_t stream);

// Kernel registration
void register_gemm_kernels();
}  // namespace cudl::kernels
