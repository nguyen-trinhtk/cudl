#include <limits>

#include "cudl/core/error.h"
#include "cudl/kernels/loss.h"
#include "cudl/kernels/registry.h"
#include "launch_utils.h"
#include "reduce_ops.cuh"

namespace cudl::kernels {
namespace {

constexpr int K_BLOCK = K_DEFAULT_BLOCK;  // num threads per block (must be mult of K_WARP)
static_assert(K_BLOCK % K_WARP == 0, "one warp per row needs whole warps");

// ---- Kernels ----

// One warp per row. Three passes over the row (max, sum of exp, gradient)
__global__ void cross_entropy_loss_warp_kernel(const float* __restrict__ logits, const int32_t* __restrict__ labels,
                                               float* __restrict__ loss, float* __restrict__ dlogits, size_t rows,
                                               size_t cols, float grad_scale) {
    const unsigned lane = threadIdx.x % K_WARP;
    const size_t num_warp = static_cast<size_t>(gridDim.x) * blockDim.x / K_WARP;
    const size_t first = (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) / K_WARP;
    for (size_t r = first; r < rows; r += num_warp) {
        const float* x = logits + r * cols;
        float* dx = dlogits + r * cols;

        float m = MaxOp::identity();
        for (size_t c = lane; c < cols; c += K_WARP) {
            m = fmaxf(m, x[c]);
        }
        m = warp_allreduce(MaxOp{}, m);

        float s = SumOp::identity();
        for (size_t c = lane; c < cols; c += K_WARP) {
            s += expf(x[c] - m);
        }
        s = warp_allreduce(SumOp{}, s);
        const float lse = m + logf(s);

        const int32_t label = labels[r];
        const bool valid = label >= 0 && static_cast<size_t>(label) < cols;
        const float nan = std::numeric_limits<float>::quiet_NaN();
        for (size_t c = lane; c < cols; c += K_WARP) {
            const float p = expf(x[c] - lse);
            const float target = static_cast<size_t>(label) == c ? 1.0f : 0.0f;
            dx[c] = valid ? (p - target) * grad_scale : nan;
        }
        if (lane == 0) {
            loss[r] = valid ? lse - x[label] : nan;
        }
    }
}

// ---- Launchers (registry signatures) ----

void cross_entropy_loss_warp(const float* logits, const int32_t* labels, float* loss, float* dlogits, size_t rows,
                             size_t cols, float grad_scale, cudaStream_t stream) {
    if (rows == 0) return;
    const unsigned grid = grid_size(rows * K_WARP, K_BLOCK);
    cross_entropy_loss_warp_kernel<<<grid, K_BLOCK, 0, stream>>>(logits, labels, loss, dlogits, rows, cols, grad_scale);
    CUDL_KERNEL_CHECK();
}
}  // namespace

// ---- Registration ----

void register_loss_kernels() {
    detail::raw_registry<CrossEntropyLossFn>().add(
        {"cross_entropy_loss", "warp", kLoadScalar, &cross_entropy_loss_warp});
}
}  // namespace cudl::kernels
