#pragma once

#include <algorithm>
#include <cstddef>

#include "cudl/core/context.h"

namespace cudl::kernels {
// Grid size for grid-stride kernels: cover `work_items`
// Cap block per SM -> huge inputs reuse threads 
// (instead of launching too many blocks)
inline unsigned grid_size(size_t work_items, int block, int blocks_per_sm = 8) {
    const size_t needed = (work_items + block - 1) / block;
    const size_t cap = static_cast<size_t>(Context::get().sm_count()) * blocks_per_sm;
    return static_cast<unsigned>(std::max<size_t>(1, std::min(needed, cap)));
}
}  // namespace cudl::kernels
