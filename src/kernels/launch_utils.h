#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "cudl/core/context.h"
#include "cudl/kernels/registry.h"

namespace cudl::kernels {
// Hardware constants
inline constexpr int K_WARP = 32; // num threads per warp
inline constexpr int K_MAX_BLOCK = 1024; // max threads per block
inline constexpr int MAX_NUM_WARP = K_MAX_BLOCK / K_WARP; // max warps per block

// Default threads per block
// each kernel file sets its own K_BLOCK from it (or overrides)
inline constexpr int K_DEFAULT_BLOCK = 256;

// Grid size for grid-stride kernels: cover `work_items`
// Cap block per SM -> huge inputs reuse threads
// (instead of launching too many blocks)
inline unsigned grid_size(size_t work_items, int block, int blocks_per_sm = 8) {
    const size_t needed = (work_items + block - 1) / block;
    const size_t cap = static_cast<size_t>(Context::get().sm_count()) * blocks_per_sm;
    return static_cast<unsigned>(std::max<size_t>(1, std::min(needed, cap)));
}

// How each thread loads its data: one float or one float4 per step
enum class Load { Scalar, Vec4 };

// Check if memory is 16-byte aligned (required for float4 loads)
inline bool aligned16(const void* p) {
    return reinterpret_cast<std::uintptr_t>(p) % 16 == 0;
}

// Registry name of a load axis value
constexpr const char* load_name(Load l) {
    return l == Load::Vec4 ? kLoadVec4 : kLoadScalar;
}

// Register the Scalar and Vec4 instantiations of one op/algo pair
template <typename Fn>
void add_loads(Registry<Fn>& r, const char* op, const char* algo, Fn scalar, Fn vec4, bool deterministic = true) {
    r.add({op, algo, load_name(Load::Scalar), scalar, deterministic});
    r.add({op, algo, load_name(Load::Vec4), vec4, deterministic});
}

// Work items per launch: each thread handles one float4 or one float
inline size_t work_items(size_t n, Load l) {
    return l == Load::Vec4 ? (n + 3) / 4 : n;
}
}  // namespace cudl::kernels
