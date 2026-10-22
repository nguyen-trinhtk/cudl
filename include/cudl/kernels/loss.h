#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace cudl::kernels {
// Cross-entropy loss with fused softmax
// Two perks: 1. numerical stability 2. less data movement
// TODO: check edge cases (NaNs)

// Variants:
// TODO try naive and smem just in case
//   warp: one warp per row (warp shuffles for the row max and sum)
using CrossEntropyLossFn = void (*)(const float* logits, const int32_t* labels, float* loss, float* dlogits,
                                    size_t rows, size_t cols, float grad_scale, cudaStream_t stream);
// Kernel registration
void register_loss_kernels();
}  // namespace cudl::kernels
