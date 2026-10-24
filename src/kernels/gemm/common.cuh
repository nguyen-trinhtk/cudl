#pragma once

#include <cstddef>
#include <type_traits>

// Ceiling division for grid and tile counts
#define CEIL_DIV(p, q) (((p) + (q) - 1) / (q))

// Shared helpers for GEMM kernels
namespace cudl::kernels::gemm_impl {

// ---- Device helpers ----
// Load or load transpose
template <bool Trans>
__device__ __forceinline__ float load_op(const float* __restrict__ X, size_t r, size_t c, size_t rows, size_t cols) {
    if constexpr (Trans) {
        return X[c * rows + r];  // col-major storage
    } else {
        return X[r * cols + c];
    }
}

// ---- Host helpers ----
template <typename Launch>
void dispatch_trans(bool trans_a, bool trans_b, Launch&& launch) {
    // Map transpose flags from runtime var to compiletime const
    // Prefer templated compiletime flags for optimized kernel execution
    if (trans_a && trans_b) {
        launch(std::true_type{}, std::true_type{});
    } else if (trans_a) {
        launch(std::true_type{}, std::false_type{});
    } else if (trans_b) {
        launch(std::false_type{}, std::true_type{});
    } else {
        launch(std::false_type{}, std::false_type{});
    }
}
}  // namespace cudl::kernels::gemm_impl
