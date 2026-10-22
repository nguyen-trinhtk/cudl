#include "cudl/core/error.h"
#include "cudl/kernels/optim.h"
#include "cudl/kernels/registry.h"
#include "launch_utils.h"

namespace cudl::kernels {
namespace {

constexpr int K_BLOCK = K_DEFAULT_BLOCK;  // num threads per block

// ---- Kernels ----

__global__ void sgd_kernel(float* __restrict__ param, const float* __restrict__ grad, float* __restrict__ velocity,
                           size_t n, float lr, float momentum) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
        float step = grad[i];
        // Uniform across the grid -> branch doesnt diverge
        if (momentum != 0.0f) {
            step = momentum * velocity[i] + step;
            velocity[i] = step;
        }
        param[i] -= lr * step;
    }
}

__global__ void adam_kernel(float* __restrict__ param, const float* __restrict__ grad, float* __restrict__ m,
                            float* __restrict__ v, size_t n, AdamStep hp) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
        const float g = grad[i];
        const float mi = hp.beta1 * m[i] + (1.0f - hp.beta1) * g;
        const float vi = hp.beta2 * v[i] + (1.0f - hp.beta2) * g * g;
        m[i] = mi;
        v[i] = vi;
        const float m_hat = mi / hp.bias_correction1;
        const float v_hat = vi / hp.bias_correction2;
        param[i] -= hp.lr * m_hat / (sqrtf(v_hat) + hp.eps);
    }
}

// ---- Launchers ----

void sgd(float* param, const float* grad, float* velocity, size_t n, float lr, float momentum, cudaStream_t stream) {
    if (n == 0) return;
    CUDL_CHECK(momentum == 0.0f || velocity != nullptr, "sgd: momentum needs a velocity buffer");
    sgd_kernel<<<grid_size(n, K_BLOCK), K_BLOCK, 0, stream>>>(param, grad, velocity, n, lr, momentum);
    CUDL_KERNEL_CHECK();
}

void adam(float* param, const float* grad, float* m, float* v, size_t n, AdamStep hp, cudaStream_t stream) {
    if (n == 0) return;
    adam_kernel<<<grid_size(n, K_BLOCK), K_BLOCK, 0, stream>>>(param, grad, m, v, n, hp);
    CUDL_KERNEL_CHECK();
}
}  // namespace

// ---- Registration ----

void register_optim_kernels() {
    detail::raw_registry<SgdFn>().add({"sgd", "", kLoadScalar, &sgd});
    detail::raw_registry<AdamFn>().add({"adam", "", kLoadScalar, &adam});
}
}  // namespace cudl::kernels
