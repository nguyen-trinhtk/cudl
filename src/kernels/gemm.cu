#include "cudl/core/error.h"
#include "cudl/kernels/gemm.h"
#include "cudl/kernels/registry.h"
#include "launch_utils.h"

namespace cudl::kernels {
namespace {
enum class Variant { Naive };

// ---- Kernels ----
template <Variant V>
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
            acc += a[row * k + i] * b[i * n + col];
        }
        float* out = &c[row * n + col];
        // beta = 0: skip reading C
        const float scaled = alpha * acc;
        *out = beta == 0.0f ? scaled : scaled + beta * (*out);
    }
}

// ---- Launchers ----
template <Variant V>
void gemm(const float* a, const float* b, float* c, size_t m, size_t n, size_t k, float alpha, float beta,
          cudaStream_t stream) {
    if (m == 0 || n == 0) return;
    const dim3 block(16, 16);
    const size_t grid_y = (m + block.y - 1) / block.y;
    CUDL_CHECK(grid_y <= 65535, "gemm: m = " << m << " exceeds the grid's y limit");
    const dim3 grid(static_cast<unsigned>((n + block.x - 1) / block.x), static_cast<unsigned>(grid_y));
    gemm_kernel<V><<<grid, block, 0, stream>>>(a, b, c, m, n, k, alpha, beta);
    CUDL_KERNEL_CHECK();
}
}  // namespace

// Kernel registration
void register_gemm_kernels() {
    detail::raw_registry<GemmFn>().add({"gemm", "naive", load_name(Load::Scalar), &gemm<Variant::Naive>});
}
}  // namespace cudl::kernels
