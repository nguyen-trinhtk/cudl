#pragma once

#include <limits>

#include "launch_utils.h"

namespace cudl::kernels {

// ---- Functors ----

// Reduction functors:
// - Identity element for empty input
// - Combine operator
// - Atomic combine
struct SumOp {
    static constexpr float identity() {
        return 0.0f;
    }
    __device__ float operator()(float a, float b) const {
        return a + b;
    }
    __device__ static void atomic(float* addr, float v) {
        atomicAdd(addr, v); // cuda atomic add
    }
};

struct MaxOp {
    static constexpr float identity() {
        return -std::numeric_limits<float>::infinity();
    }
    __device__ float operator()(float a, float b) const {
        return fmaxf(a, b);
    }

    // Compare-and-swap until stored value >= v
    // atomic max always != NaNs
    __device__ static void atomic(float* addr, float v) {
        int* bits = reinterpret_cast<int*>(addr);
        float cur = *reinterpret_cast<volatile float*>(addr);
        while (cur < v) {
            const int prev = atomicCAS(bits, __float_as_int(cur), __float_as_int(v));
            if (prev == __float_as_int(cur)) {
                break;
            }
            cur = __int_as_float(prev);
        }
    }
};

// ---- Device helpers ----

// Butterfly warp reduce: every lane ends up with the full warp's result
template <typename Op>
__device__ __forceinline__ float warp_allreduce(Op op, float v) {
    for (int offset = K_WARP / 2; offset > 0; offset >>= 1) {
        v = op(v, __shfl_xor_sync(0xffffffffu, v, offset));
    }
    return v;
}
}  // namespace cudl::kernels
