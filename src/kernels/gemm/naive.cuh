#pragma once

#include "common.cuh"
#include "cudl/core/error.h"

// Naive kernel: each thread computes one C[row, col]
namespace cudl::kernels::gemm_impl {

static const int BLOCK_SIZE_X = 16;
static const int BLOCK_SIZE_Y = 16;

template <bool TransA, bool TransB>
__global__ void naive_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c,
                             size_t m, size_t n, size_t k, float alpha, float beta) {
    // for memory coalescing, y-dim = row and x-dim = col
    const size_t row = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t col = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) 
        return;
    
    // dot product loop
    float acc = 0.0f;
    for (size_t i = 0; i < k; ++i) {
        const float av = load_op<TransA>(a, row, i, m, k);
        const float bv = load_op<TransB>(b, i, col, k, n);
        acc += av * bv;
    }
    
    float* out = &c[row * n + col];
    // beta = 0: skip reading C
    const float scaled = alpha * acc; // alpha A*B
    *out = (beta == 0.0f) ? scaled : scaled + beta * (*out);
}

inline void gemm_naive(const float* a, const float* b, float* c, size_t m, size_t n, size_t k, bool trans_a,
                       bool trans_b, float alpha, float beta, cudaStream_t stream) {
    if (m == 0 || n == 0) 
        return;
    
    const dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    
    const size_t grid_x = (n + BLOCK_SIZE_X - 1) / (BLOCK_SIZE_X);
    const size_t grid_y = (m + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y;
    CUDL_CHECK(grid_y <= 65535, "gemm: m = " << m << " exceeds the grid's y limit");
    
    const dim3 grid(static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y));
    
    const size_t smem_bytes = 0; // no dynamic smem 

    dispatch_trans(trans_a, trans_b, [&](auto ta, auto tb) {
        naive_kernel<decltype(ta)::value, decltype(tb)::value><<<grid, block, smem_bytes, stream>>>(a, b, c, m, n, k, alpha, beta);
    });

    CUDL_KERNEL_CHECK();
}
}  // namespace cudl::kernels::gemm_impl
