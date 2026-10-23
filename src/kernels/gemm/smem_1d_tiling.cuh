#pragma once

#include "common.cuh"
#include "cudl/core/error.h"

// Similar to smem-blocking but now 1 thread computes 8 slots in C for enhanced AI
// these entries are same column, consecutive rows
namespace cudl::kernels::gemm_impl {

// TODO: should we?
#define CEIL_DIV(p, q) (((p) + (q) - 1)/(q))

// Convention of M, N, K correspond to matrix dimensions
static const int BM = 64; // block size in M's dim
static const int BN = 64; // block size in N's dim
static const int BK = 8; // block size in K's dim
static const int TM = 8; // number of entries of C one thread will update

template <bool TransA, bool TransB>
__global__ void smem_1d_tiling(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                             size_t m, size_t n, size_t k, float alpha, float beta) {
    __shared__ float tile_A[BK][BM];
    __shared__ float tile_B[BK][BN];

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;

    const int by = blockIdx.y;
    const int bx = blockIdx.x;

    const size_t row = by * BM + ty * TM; // row base
    const size_t col = bx * BN + tx;

    float acc[TM] = {0.0f}; // acc is now vec8
    // Sliding tiles over K
    const int num_tiles = CEIL_DIV(k, BK);
    for (size_t b = 0; b < num_tiles; b++) {
        // tx, ty got swapped for A's tiles
        const size_t A_row = by * BM + tx;
        const size_t A_col = b * BK + ty;

        // B tile stays the same
        const size_t B_row = b * BK + ty;
        const size_t B_col = bx * BN + tx;

        // Check if the thread is in or not
        const bool a_in = (A_row < m) && (A_col < k);
        const bool b_in = (B_row < k) && (B_col < n);

        // Load tile into smem
        tile_A[ty][tx] = a_in ? load_op<TransA>(A, A_row, A_col, m, k) : 0.0f;
        tile_B[ty][tx] = b_in ? load_op<TransB>(B, B_row, B_col, k, n) : 0.0f;

        __syncthreads();

        // For each element in the edge of 
        for (size_t i = 0; i < BK; i++) {
            const float b_val = tile_B[i][tx];
            for (size_t j = 0; j < TM; j++) {
                acc[j] += tile_A[i][j + TM * ty] * b_val;
            }
        }
        __syncthreads();
    }
    // Discard boundary threads
    if (col >= n) 
        return;

    // Row offset
    for (size_t j = 0; j < TM; j++) {
        if (row + j >= m) 
            continue;
        // update C
        float* out = &C[(row + j) * n + col];
        // beta = 0: skip reading C
        const float scaled = alpha * acc[j]; // alpha A*B
        *out = (beta == 0.0f) ? scaled : scaled + beta * (*out);
    }
}


// TODO: update
inline void gemm_smem_1d_tiling(const float* A, const float* B, float* C, size_t m, size_t n, size_t k, bool trans_a,
                       bool trans_b, float alpha, float beta, cudaStream_t stream) {
    if (m == 0 || n == 0) 
        return;
    
    // one thread per C element, one block per TILE_SIZE x TILE_SIZE tile of C
    const dim3 block(BN, BM/TM); // todo: BM/TM is valid

    const size_t grid_x = CEIL_DIV(n, BN);
    const size_t grid_y = CEIL_DIV(m, BM);
    CUDL_CHECK(grid_y <= 65535, "gemm: m = " << m << " exceeds the grid's y limit");

    const dim3 grid(static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y));

    const size_t smem_bytes = 0; // tiles are static __shared__ arrays

    dispatch_trans(trans_a, trans_b, [&](auto ta, auto tb) {
        smem_1d_tiling<decltype(ta)::value, decltype(tb)::value><<<grid, block, smem_bytes, stream>>>(A, B, C, m, n, k, alpha, beta);
    });
    CUDL_KERNEL_CHECK();
}
}  // namespace cudl::kernels::gemm_impl
