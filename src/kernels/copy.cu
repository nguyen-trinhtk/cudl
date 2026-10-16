#include <cstdint>

#include "cudl/core/error.h"
#include "cudl/kernels/copy.h"
#include "cudl/kernels/registry.h"
#include "launch_utils.h"

namespace cudl::kernels {
namespace {

constexpr int kBlock = 256;

__global__ void copy_naive_kernel(const float* __restrict__ src, float* __restrict__ dst, size_t n) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) dst[i] = src[i];
}

// src and dst must be 16-byte aligned. The last n % 4 elements are copied by the first threads.
__global__ void copy_vec4_kernel(const float4* __restrict__ src, float4* __restrict__ dst, size_t n) {
    const size_t n4 = n / 4;
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = tid; i < n4; i += stride) dst[i] = src[i];

    const size_t tail = n - n4 * 4;
    if (tid < tail) {
        reinterpret_cast<float*>(dst)[n4 * 4 + tid] = reinterpret_cast<const float*>(src)[n4 * 4 + tid];
    }
}

bool aligned16(const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 16 == 0; }
}  // namespace

void copy_naive(const float* src, float* dst, size_t n, cudaStream_t stream) {
    if (n == 0) return;
    copy_naive_kernel<<<grid_size(n, kBlock), kBlock, 0, stream>>>(src, dst, n);
    CUDL_KERNEL_CHECK();
}

void copy_vec4(const float* src, float* dst, size_t n, cudaStream_t stream) {
    if (n == 0) return;
    if (!aligned16(src) || !aligned16(dst)) {
        copy_naive(src, dst, n, stream);
        return;
    }
    copy_vec4_kernel<<<grid_size((n + 3) / 4, kBlock), kBlock, 0, stream>>>(reinterpret_cast<const float4*>(src),
                                                                            reinterpret_cast<float4*>(dst), n);
    CUDL_KERNEL_CHECK();
}

void register_copy_kernels() {
    auto& r = detail::raw_registry<CopyFn>();
    r.add({"copy", "naive", &copy_naive});
    r.add({"copy", "vec4", &copy_vec4});
}
}  // namespace cudl::kernels
