#include "cudl/core/error.h"
#include "cudl/kernels/gemm.h"
#include "cudl/kernels/registry.h"
#include "launch_utils.h"

namespace cudl::kernels {
namespace {
enum class Variant { Naive };

// ---- Device helpers ----

// Load or load transpose
template <bool Trans>
__device__ __forceinline__ float load_op(const float* __restrict__ x, size_t r, size_t c, size_t rows, size_t cols) {
    if constexpr (Trans) {
        return x[c * rows + r];  // col-major storage
    } else {
        return x[r * cols + c];
    }
}

// ---- Kernels ----
template <Variant V, bool TransA, bool TransB>
// TODO: other kernels
// Only naive kernel for now
__global__ void gemm_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c,
                            size_t m, size_t n, size_t k, float alpha, float beta) {
    // for memory coalescing, y-dim = row and x-dim = col
    const size_t row = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t col = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;
    if constexpr (V == Variant::Naive) {
        float acc = 0.0f;
        for (size_t i = 0; i < k; ++i) {
            const float av = load_op<TransA>(a, row, i, m, k);
            const float bv = load_op<TransB>(b, i, col, k, n);
            acc += av * bv;
        }
        float* out = &c[row * n + col];
        // beta = 0: skip reading C
        const float scaled = alpha * acc;
        *out = beta == 0.0f ? scaled : scaled + beta * (*out);
    }
}

// ---- Host helpers ----

// Launch the (TransA, TransB) instantiation picked at run time
template <Variant V>
void launch_gemm(bool trans_a, bool trans_b, dim3 grid, dim3 block, const float* a, const float* b, float* c,
                 size_t m, size_t n, size_t k, float alpha, float beta, cudaStream_t stream) {
    if (trans_a && trans_b) {
        gemm_kernel<V, true, true><<<grid, block, 0, stream>>>(a, b, c, m, n, k, alpha, beta);
    } else if (trans_a) {
        gemm_kernel<V, true, false><<<grid, block, 0, stream>>>(a, b, c, m, n, k, alpha, beta);
    } else if (trans_b) {
        gemm_kernel<V, false, true><<<grid, block, 0, stream>>>(a, b, c, m, n, k, alpha, beta);
    } else {
        gemm_kernel<V, false, false><<<grid, block, 0, stream>>>(a, b, c, m, n, k, alpha, beta);
    }
    CUDL_KERNEL_CHECK();
}

// ---- Launchers ----
template <Variant V>
void gemm(const float* a, const float* b, float* c, size_t m, size_t n, size_t k, bool trans_a, bool trans_b,
          float alpha, float beta, cudaStream_t stream) {
    if (m == 0 || n == 0) return;
    const dim3 block(16, 16);
    const size_t grid_y = (m + block.y - 1) / block.y;
    CUDL_CHECK(grid_y <= 65535, "gemm: m = " << m << " exceeds the grid's y limit");
    const dim3 grid(static_cast<unsigned>((n + block.x - 1) / block.x), static_cast<unsigned>(grid_y));
    launch_gemm<V>(trans_a, trans_b, grid, block, a, b, c, m, n, k, alpha, beta, stream);
}
}  // namespace

// Kernel registration
void register_gemm_kernels() {
    detail::raw_registry<GemmFn>().add({"gemm", "naive", load_name(Load::Scalar), &gemm<Variant::Naive>});
}
}  // namespace cudl::kernels
