#pragma once

#include "common.cuh"
#include "../launch_utils.h"
#include "cudl/core/error.h"

// Similar to 1D tiling but now 1 thread computes a TMxTN tile of C
// so each smem value of A and B is reused across a whole row/column of that tile
// With option for vectorized load/store when possible
namespace cudl::kernels::gemm_impl {

// Convention of M, N, K correspond to matrix dimensions
// BM, BN, BK: block tile sizes in M, N, K's dim
// TM, TN: rows, cols of C per thread
// L: Scalar loads 1 float per step, Vec4 loads/stores float4
template <int BM, int BN, int BK, int TM, int TN, Load L, bool TransA, bool TransB>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
smem_2d_tiling(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
               size_t m, size_t n, size_t k, float alpha, float beta) {
    constexpr int NUM_THREADS = (BM / TM) * (BN / TN);

    // A is transposed, again check bank conflict
    __shared__ __align__(16) float tile_A[BK][BM];
    __shared__ __align__(16) float tile_B[BK][BN];

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
        if constexpr (L == Load::Vec4) {
            // Vec4 loading, for now only support untransposed
            static_assert(!TransA && !TransB, "Vec4 path reads untransposed rows");
            
            const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            
            // for now only accept k % 4 == 0 and n % 4 == 0
            // TODO: scalarize the remainders

            #pragma unroll
            for (int i = 0; i < BM * BK / 4 / NUM_THREADS; i++) {
                const int idx = i * NUM_THREADS + tid; // float4 index
                const int tA_row = idx / (BK / 4);
                const int tA_col = idx % (BK / 4) * 4;

                const size_t A_row = by * BM + tA_row;
                const size_t A_col = b * BK + tA_col;

                const bool a_in = (A_row < m) && (A_col < k);
                const float4 v = a_in ? *reinterpret_cast<const float4*>(&A[A_row * k + A_col]) : zero;
                
                // transpose on the way into smem
                tile_A[tA_col + 0][tA_row] = v.x;
                tile_A[tA_col + 1][tA_row] = v.y;
                tile_A[tA_col + 2][tA_row] = v.z;
                tile_A[tA_col + 3][tA_row] = v.w;
            }

            #pragma unroll
            for (int i = 0; i < BK * BN / 4 / NUM_THREADS; i++) {
                const int idx = i * NUM_THREADS + tid; // float4 index
                
                const int tB_row = idx / (BN / 4);
                const int tB_col = idx % (BN / 4) * 4;
                
                const size_t B_row = b * BK + tB_row;
                const size_t B_col = bx * BN + tB_col;
                
                const bool b_in = (B_row < k) && (B_col < n);
                const float4 v = b_in ? *reinterpret_cast<const float4*>(&B[B_row * n + B_col]) : zero;
                
                *reinterpret_cast<float4*>(&tile_B[tB_row][tB_col]) = v;
            }
        } else {
            #pragma unroll
            for (int i = 0; i < BM * BK / NUM_THREADS; i++) {
                const int idx = i * NUM_THREADS + tid;
                const int tA_row = idx / BK;
                const int tA_col = idx % BK;
                const size_t A_row = by * BM + tA_row;
                const size_t A_col = b * BK + tA_col;
                const bool a_in = (A_row < m) && (A_col < k);
                tile_A[tA_col][tA_row] = a_in ? load_op<TransA>(A, A_row, A_col, m, k) : 0.0f; // transposed
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
        }
        __syncthreads();

        // Compute
        // Unroll these tiny loops so acc stays in registers
        // and the compiler can hoist smem loads out of inner loops
        #pragma unroll
        for (int ck = 0; ck < BK; ck++) {
            if constexpr (L == Load::Vec4) {
                // Stage the thread's column of A and row of B in registers, float4 at a time
                float reg_A[TM];
                float reg_B[TN];
                #pragma unroll
                for (int i = 0; i < TM; i += 4) {
                    const float4 v = *reinterpret_cast<const float4*>(&tile_A[ck][ty * TM + i]);
                    reg_A[i + 0] = v.x;
                    reg_A[i + 1] = v.y;
                    reg_A[i + 2] = v.z;
                    reg_A[i + 3] = v.w;
                }
                #pragma unroll
                for (int i = 0; i < TN; i += 4) {
                    const float4 v = *reinterpret_cast<const float4*>(&tile_B[ck][tx * TN + i]);
                    reg_B[i + 0] = v.x;
                    reg_B[i + 1] = v.y;
                    reg_B[i + 2] = v.z;
                    reg_B[i + 3] = v.w;
                }
                #pragma unroll
                for (int cy = 0; cy < TM; cy++) {
                    #pragma unroll
                    for (int cx = 0; cx < TN; cx++) {
                        acc[cy][cx] += reg_A[cy] * reg_B[cx];
                    }
                }
            } else {
                #pragma unroll
                for (int cy = 0; cy < TM; cy++) {
                    #pragma unroll
                    for (int cx = 0; cx < TN; cx++) {
                        acc[cy][cx] += tile_A[ck][ty * TM + cy] * tile_B[ck][tx * TN + cx];
                    }
                }
            }
        }
        __syncthreads();
    }

    // Store from acc to C
    const size_t row = by * BM + ty * TM;
    const size_t col = bx * BN + tx * TN;

    if constexpr (L == Load::Vec4) {
        // n % 4 == 0, so each float4 of C is fully in or out
        #pragma unroll
        for (int r = 0; r < TM; r++) { // row offset
            #pragma unroll
            for (int c = 0; c < TN; c += 4) { // col offset
                if (row + r >= m || col + c >= n) {
                    continue;
                }
                float4* out = reinterpret_cast<float4*>(&C[(row + r) * n + (col + c)]);
                float4 v = make_float4(alpha * acc[r][c], alpha * acc[r][c + 1], alpha * acc[r][c + 2],
                                       alpha * acc[r][c + 3]);
                // beta = 0: skip reading C
                if (beta != 0.0f) {
                    const float4 prev = *out;
                    v.x += beta * prev.x;
                    v.y += beta * prev.y;
                    v.z += beta * prev.z;
                    v.w += beta * prev.w;
                }
                *out = v;
            }
        }
    } else {
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
}


template <int BM, int BN, int BK, int TM, int TN, Load L>
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

    if constexpr (L == Load::Vec4) {
        static_assert(BK % 4 == 0 && BN % 4 == 0 && TM % 4 == 0 && TN % 4 == 0, "Vec4 needs tile sizes divisible by 4");
        static_assert((BM * BK / 4) % NUM_THREADS == 0 && (BK * BN / 4) % NUM_THREADS == 0,
                      "float4 tiles must split evenly over threads");
        // float4 needs untransposed inputs, 16B-aligned pointers and rows (k, n multiple of 4)
        // otherwise fall back to scalar
        const bool untransposed = !trans_a && !trans_b;
        const bool rows_aligned = (k % 4 == 0) && (n % 4 == 0);
        const bool ptrs_aligned = aligned16(A) && aligned16(B) && aligned16(C);
        if (untransposed && rows_aligned && ptrs_aligned) {
            smem_2d_tiling<BM, BN, BK, TM, TN, Load::Vec4, false, false>
                <<<grid, block, smem_bytes, stream>>>(A, B, C, m, n, k, alpha, beta);
            CUDL_KERNEL_CHECK();
            return;
        }
    }

    dispatch_trans(trans_a, trans_b, [&](auto ta, auto tb) {
        smem_2d_tiling<BM, BN, BK, TM, TN, Load::Scalar, decltype(ta)::value, decltype(tb)::value>
            <<<grid, block, smem_bytes, stream>>>(A, B, C, m, n, k, alpha, beta);
    });
    CUDL_KERNEL_CHECK();
}
}  // namespace cudl::kernels::gemm_impl
