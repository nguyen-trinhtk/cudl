#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace cudl::kernels {

// Shared signature for copy functions
using CopyFn = void (*)(const float* src, float* dst, size_t n, cudaStream_t stream);

// Naive: Copy one element per thread
void copy_naive(const float* src, float* dst, size_t n, cudaStream_t stream);
// Vec4: copy 4 floats at once (16 bytes), fallback to naive for misaligned chunks
void copy_vec4(const float* src, float* dst, size_t n, cudaStream_t stream);
void register_copy_kernels();
}  // namespace cudl::kernels
