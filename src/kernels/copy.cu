#include "cudl/core/error.h"
#include "cudl/kernels/copy.h"
#include "cudl/kernels/registry.h"
#include "launch_utils.h"

namespace cudl::kernels {
namespace {

constexpr int K_BLOCK = K_DEFAULT_BLOCK;  // num threads per block

// ---- Kernels ----

// Vec4: src and dst must be 16-byte aligned. The last n % 4 elements are copied by the first threads.
template <Load L>
__global__ void copy_kernel(const float* __restrict__ src, float* __restrict__ dst, size_t n) {
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    if constexpr (L == Load::Vec4) {
        const float4* src4 = reinterpret_cast<const float4*>(src);
        float4* dst4 = reinterpret_cast<float4*>(dst);
        const size_t n4 = n / 4;
        for (size_t i = tid; i < n4; i += stride) dst4[i] = src4[i];

        const size_t tail = n - n4 * 4;
        if (tid < tail) {
            dst[n4 * 4 + tid] = src[n4 * 4 + tid];
        }
    } else {
        for (size_t i = tid; i < n; i += stride) dst[i] = src[i];
    }
}

// ---- Host helpers ----

// Launch the Load instantiation picked at run time
void launch_copy(Load l, const float* src, float* dst, size_t n, cudaStream_t stream) {
    const unsigned grid = grid_size(work_items(n, l), K_BLOCK);
    if (l == Load::Vec4) {
        copy_kernel<Load::Vec4><<<grid, K_BLOCK, 0, stream>>>(src, dst, n);
    } else {
        copy_kernel<Load::Scalar><<<grid, K_BLOCK, 0, stream>>>(src, dst, n);
    }
    CUDL_KERNEL_CHECK();
}

// ---- Launchers (registry signatures) ----

template <Load L>
void copy(const float* src, float* dst, size_t n, cudaStream_t stream) {
    if (n == 0) return;
    // use vec4 kernel only if data is 16byte aligned
    const bool vec4 = (L == Load::Vec4 && aligned16(src) && aligned16(dst));
    launch_copy(vec4 ? Load::Vec4 : Load::Scalar, src, dst, n, stream);
}
}  // namespace

// ---- Registration ----

void register_copy_kernels() {
    add_loads(detail::raw_registry<CopyFn>(), "copy", "", &copy<Load::Scalar>, &copy<Load::Vec4>);
}
}  // namespace cudl::kernels
