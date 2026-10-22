#pragma once

#include "common.cuh"
#include "cudl/core/error.h"

// Smem cache-blocking kernel with 32x32 tiles
// Each thread still compute 1 element in C
namespace cudl::kernels::gemm_impl {

static const int TILE_SIZE = 32;

template <bool TransA, bool TransB>
__global__ void smem_blocking(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                             size_t m, size_t n, size_t k, float alpha, float beta) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int by = blockIdx.y;
    const int bx = blockIdx.x;

    const size_t row = by * TILE_SIZE + ty; // y
    const size_t col = bx * TILE_SIZE + tx; // x
    
    const size_t NUM_TILES = (k + TILE_SIZE - 1) / TILE_SIZE;

    float acc = 0.0f;
    for (size_t b = 0; b < NUM_TILES; b++) {
        // For this TILE
        const size_t A_col = b * TILE_SIZE + tx;
        const size_t B_row = b * TILE_SIZE + ty;

        // Check if the thread is in or not
        const bool a_in = (row < m) && (A_col < k);
        const bool b_in = (B_row < k) && (col < n);

        // Load into smem, = 0 if out-bound
        tile_A[ty][tx] = a_in ? load_op<TransA>(A, row, A_col, m, k) : 0.0f;
        tile_B[ty][tx] = b_in ? load_op<TransB>(B, B_row, col, k, n) : 0.0f;
        __syncthreads();
        for (size_t i = 0; i < TILE_SIZE; i++) {
            acc += tile_A[ty][i] * tile_B[i][tx];
        }
        __syncthreads(); // sync before next write
    }
    // Discard boundary threads
    if (row >= m || col >= n) 
        return;

    float* out = &C[row * n + col];
    // beta = 0: skip reading C
    const float scaled = alpha * acc; // alpha A*B
    *out = (beta == 0.0f) ? scaled : scaled + beta * (*out);
}

inline void gemm_smem_blocking(const float* A, const float* B, float* C, size_t m, size_t n, size_t k, bool trans_a,
                       bool trans_b, float alpha, float beta, cudaStream_t stream) {
    if (m == 0 || n == 0) 
        return;
    
    // one thread per C element, one block per TILE_SIZE x TILE_SIZE tile of C
    const dim3 block(TILE_SIZE, TILE_SIZE);

    const size_t grid_x = (n + TILE_SIZE - 1) / TILE_SIZE;
    const size_t grid_y = (m + TILE_SIZE - 1) / TILE_SIZE;
    CUDL_CHECK(grid_y <= 65535, "gemm: m = " << m << " exceeds the grid's y limit");

    const dim3 grid(static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y));

    const size_t smem_bytes = 0; // tiles are static __shared__ arrays

    dispatch_trans(trans_a, trans_b, [&](auto ta, auto tb) {
        smem_blocking<decltype(ta)::value, decltype(tb)::value><<<grid, block, smem_bytes, stream>>>(A, B, C, m, n, k, alpha, beta);
    });
    CUDL_KERNEL_CHECK();
}
}  // namespace cudl::kernels::gemm_impl
