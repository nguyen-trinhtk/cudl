#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace cudl::kernels {

// Shared signature for copy functions
// Variants:
//   scalar: copy one element per thread
//   vec4: copy 4 floats at once (16 bytes), fallback to scalar if src or dst is misaligned
using CopyFn = void (*)(const float* src, float* dst, size_t n, cudaStream_t stream);

// Kernel registration
void register_copy_kernels();
}  // namespace cudl::kernels
