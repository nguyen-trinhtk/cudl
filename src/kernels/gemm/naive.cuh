#pragma once

#include "common.cuh"
#include "cudl/core/error.h"

// Naive kernel: each thread computes one C[row, col]
namespace cudl::kernels::gemm_impl {

// BX, BY: block size in x and y
// Coalesced: x = col, y = row, so a warp reads/writes neighbouring elements of B and C
// otherwise x = row, y = col, so a warp's accesses to A and C are a whole row apart (no coalescing)
template <int BX, int BY, bool Coalesced, bool TransA, bool TransB>
__global__ void __launch_bounds__(BX * BY)
naive_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                             size_t m, size_t n, size_t k, float alpha, float beta) {
    const size_t gx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t gy = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t row = Coalesced ? gy : gx;
    const size_t col = Coalesced ? gx : gy;
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

template <int BX, int BY, bool Coalesced = true>
inline void gemm_naive(const float* A, const float* B, float* C, size_t m, size_t n, size_t k, bool trans_a,
                       bool trans_b, float alpha, float beta, cudaStream_t stream) {
    static_assert(BX * BY <= 1024, "too many threads per block");

    if (m == 0 || n == 0) 
        return;
    
    const dim3 block(BX, BY);
    
    // x covers cols (n) when coalesced, rows (m) otherwise
    const size_t grid_x = CEIL_DIV(Coalesced ? n : m, BX);
    const size_t grid_y = CEIL_DIV(Coalesced ? m : n, BY);
    CUDL_CHECK(grid_y <= 65535, "gemm: " << (Coalesced ? "m = " : "n = ") << (Coalesced ? m : n)
                                         << " exceeds the grid's y limit");
    
    const dim3 grid(static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y));
    
    const size_t smem_bytes = 0; // no dynamic smem 

    dispatch_trans(trans_a, trans_b, [&](auto ta, auto tb) {
        naive_kernel<BX, BY, Coalesced, decltype(ta)::value, decltype(tb)::value><<<grid, block, smem_bytes, stream>>>(A, B, C, m, n, k, alpha, beta);
    });

    CUDL_KERNEL_CHECK();
}
}  // namespace cudl::kernels::gemm_impl
