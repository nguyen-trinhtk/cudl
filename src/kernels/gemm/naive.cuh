#pragma once

#include "common.cuh"
#include "cudl/core/error.h"

// Naive kernel: each thread computes one C[row, col]
namespace cudl::kernels::gemm_impl {

// BX, BY: block size in x (cols of C) and y (rows of C)
template <int BX, int BY, bool TransA, bool TransB>
__global__ void __launch_bounds__(BX * BY)
naive_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                             size_t m, size_t n, size_t k, float alpha, float beta) {
    // for memory coalescing, y-dim = row and x-dim = col
    const size_t row = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t col = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) 
        return;
    
    // dot product loop
    float acc = 0.0f;
    for (size_t i = 0; i < k; ++i) {
        const float av = load_op<TransA>(A, row, i, m, k);
        const float bv = load_op<TransB>(B, i, col, k, n); // not coalesced if B is transposed :(
        acc += av * bv;
    }
    
    float* out = &C[row * n + col];
    // beta = 0: skip reading C
    const float scaled = alpha * acc; // alpha A*B
    *out = (beta == 0.0f) ? scaled : scaled + beta * (*out);
}

template <int BX, int BY>
inline void gemm_naive(const float* A, const float* B, float* C, size_t m, size_t n, size_t k, bool trans_a,
                       bool trans_b, float alpha, float beta, cudaStream_t stream) {
    static_assert(BX * BY <= 1024, "too many threads per block");

    if (m == 0 || n == 0) 
        return;
    
    const dim3 block(BX, BY);
    
    const size_t grid_x = CEIL_DIV(n, BX);
    const size_t grid_y = CEIL_DIV(m, BY);
    CUDL_CHECK(grid_y <= 65535, "gemm: m = " << m << " exceeds the grid's y limit");
    
    const dim3 grid(static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y));
    
    const size_t smem_bytes = 0; // no dynamic smem 

    dispatch_trans(trans_a, trans_b, [&](auto ta, auto tb) {
        naive_kernel<BX, BY, decltype(ta)::value, decltype(tb)::value><<<grid, block, smem_bytes, stream>>>(A, B, C, m, n, k, alpha, beta);
    });

    CUDL_KERNEL_CHECK();
}
}  // namespace cudl::kernels::gemm_impl
