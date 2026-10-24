#pragma once

#include "common.cuh"
#include "cudl/core/error.h"

// Similar to 1D tiling but now 1 thread computes a TMxTN tile of C
// so each smem value of A and B is reused across a whole row/column of that tile
namespace cudl::kernels::gemm_impl {

// Convention of M, N, K correspond to matrix dimensions
// BM, BN, BK: block tile sizes in M, N, K's dim
// TM, TN: rows, cols of C per thread
template <int BM, int BN, int BK, int TM, int TN, bool TransA, bool TransB>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
smem_2d_tiling(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
               size_t m, size_t n, size_t k, float alpha, float beta) {
    constexpr int NUM_THREADS = (BM / TM) * (BN / TN);

    __shared__ float tile_A[BM][BK]; // TODO: transpose/pad this to avoid conflict?
    __shared__ float tile_B[BK][BN];

    const int by = blockIdx.y;
    const int bx = blockIdx.x;
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int tid = ty * (BN / TN) + tx; // flat id, only used for loading

    float acc[TM][TN] = {0.0f};
    // Sliding tiles over K
    const size_t num_tiles = CEIL_DIV(k, BK);
    for (size_t b = 0; b < num_tiles; b++) {
        // Load: flattened id for coalescing
        #pragma unroll
        for (int i = 0; i < BM * BK / NUM_THREADS; i++) {
            const int idx = i * NUM_THREADS + tid;
            const int tA_row = idx / BK;
            const int tA_col = idx % BK;
            const size_t A_row = by * BM + tA_row;
            const size_t A_col = b * BK + tA_col;
            const bool a_in = (A_row < m) && (A_col < k);
            tile_A[tA_row][tA_col] = a_in ? load_op<TransA>(A, A_row, A_col, m, k) : 0.0f;
        }
        #pragma unroll
        for (int i = 0; i < BK * BN / NUM_THREADS; i++) {
            const int idx = i * NUM_THREADS + tid;
            const int tB_row = idx / BN;
            const int tB_col = idx % BN;
            const size_t B_row = b * BK + tB_row;
            const size_t B_col = bx * BN + tB_col;
            const bool b_in = (B_row < k) && (B_col < n);
            tile_B[tB_row][tB_col] = b_in ? load_op<TransB>(B, B_row, B_col, k, n) : 0.0f;
        }
        __syncthreads();

        // Compute
        // Unroll these tiny loops so acc stays in registers
        // and the compiler can hoist the smem loads out of the inner loops
        #pragma unroll
        for (int ck = 0; ck < BK; ck++) {
            #pragma unroll
            for (int cy = 0; cy < TM; cy++) {
                #pragma unroll
                for (int cx = 0; cx < TN; cx++) {
                    acc[cy][cx] += tile_A[ty * TM + cy][ck] * tile_B[ck][tx * TN + cx];
                }
            }
        }
        __syncthreads();
    }

    // Store from acc to C
    const size_t row = by * BM + ty * TM;
    const size_t col = bx * BN + tx * TN;

    #pragma unroll
    for (int r = 0; r < TM; r++) { // row offset
        #pragma unroll
        for (int c = 0; c < TN; c++) { // col offset
            if (row + r >= m || col + c >= n) {
                continue;
            }
            float* out = &C[(row + r) * n + (col + c)];
            // beta = 0: skip reading C
            const float scaled = alpha * acc[r][c];
            *out = (beta == 0.0f) ? scaled : scaled + beta * (*out);
        }
    }
}


template <int BM, int BN, int BK, int TM, int TN>
inline void gemm_smem_2d_tiling(const float* A, const float* B, float* C, size_t m, size_t n, size_t k, bool trans_a,
                       bool trans_b, float alpha, float beta, cudaStream_t stream) {
    constexpr int NUM_THREADS = (BM / TM) * (BN / TN);
    static_assert(BM % TM == 0 && BN % TN == 0, "thread tile must divide block tile");
    static_assert((BM * BK) % NUM_THREADS == 0 && (BK * BN) % NUM_THREADS == 0, "smem tiles must split evenly over threads");
    static_assert(NUM_THREADS <= 1024, "too many threads per block");
    static_assert((BM * BK + BK * BN) * sizeof(float) <= 48 * 1024, "tiles exceed the 48KB static smem limit");

    if (m == 0 || n == 0)
        return;

    // one thread per TMxTN tile of C, one block per BMxBN tile of C
    const dim3 block(BN / TN, BM / TM);

    const size_t grid_x = CEIL_DIV(n, BN);
    const size_t grid_y = CEIL_DIV(m, BM);
    CUDL_CHECK(grid_y <= 65535, "gemm: m = " << m << " exceeds the grid's y limit");

    const dim3 grid(static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y));

    const size_t smem_bytes = 0; // tiles are static __shared__ arrays

    dispatch_trans(trans_a, trans_b, [&](auto ta, auto tb) {
        smem_2d_tiling<BM, BN, BK, TM, TN, decltype(ta)::value, decltype(tb)::value>
            <<<grid, block, smem_bytes, stream>>>(A, B, C, m, n, k, alpha, beta);
    });
    CUDL_KERNEL_CHECK();
}
}  // namespace cudl::kernels::gemm_impl
