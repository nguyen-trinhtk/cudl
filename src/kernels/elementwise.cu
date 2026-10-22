#include "cudl/core/error.h"
#include "cudl/kernels/elementwise.h"
#include "cudl/kernels/registry.h"
#include "elementwise_ops.cuh"
#include "launch_utils.h"

namespace cudl::kernels {
namespace {

constexpr int K_BLOCK = K_DEFAULT_BLOCK;  // num threads per block

// Input pointers for an N-ary op, passed to the kernel by value.
template <int N>
struct Inputs {
    const float* p[N];
};

// ---- Device helpers ----
// (functors live in elementwise_ops.cuh)

// Unary
template <typename Op>
__device__ __forceinline__ float apply(const Op& op, const Inputs<1>& in, size_t i) {
    return op(in.p[0][i]);
}

// Binary
template <typename Op>
__device__ __forceinline__ float apply(const Op& op, const Inputs<2>& in, size_t i) {
    return op(in.p[0][i], in.p[1][i]);
}

// Vectorized: Same as apply, but on the i-th float4 of each input
// Unary
template <typename Op>
__device__ __forceinline__ float4 apply4(const Op& op, const Inputs<1>& in, size_t i) {
    const float4 a = reinterpret_cast<const float4*>(in.p[0])[i]; // float4
    return {op(a.x), op(a.y), op(a.z), op(a.w)};
}

// Binary
template <typename Op>
__device__ __forceinline__ float4 apply4(const Op& op, const Inputs<2>& in, size_t i) {
    const float4 a = reinterpret_cast<const float4*>(in.p[0])[i];
    const float4 b = reinterpret_cast<const float4*>(in.p[1])[i];
    return {op(a.x, b.x), op(a.y, b.y), op(a.z, b.z), op(a.w, b.w)};
}

// ---- Kernels ----

// Template in op & num input per index (to support both unary and binary)
// Vec4: float4 loads/stores, all pointers must be 16-byte aligned
template <Load L, typename Op, int N>
__global__ void elementwise_kernel(Op op, Inputs<N> in, float* __restrict__ out, size_t n) {
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    if constexpr (L == Load::Vec4) {
        // Apply 4
        const size_t n4 = n / 4;
        for (size_t i = tid; i < n4; i += stride) {
            reinterpret_cast<float4*>(out)[i] = apply4(op, in, i);
        }
        // Last n % 4 elements are handled by the first threads
        const size_t tail = n - n4 * 4;
        if (tid < tail) { // remaining elements
            out[n4 * 4 + tid] = apply(op, in, n4 * 4 + tid);
        }
    } else {
        for (size_t i = tid; i < n; i += stride) {
            out[i] = apply(op, in, i);
        }
    }
}

// Bias kernel
// Vec4: requires cols % 4 == 0 and 16-byte aligned pointers
// so a float4 never straddles two rows
template <Load L, typename Op>
__global__ void bias_kernel(
    Op op, const float* __restrict__ x, const float* __restrict__ bias,
    float* __restrict__ out, size_t n, size_t cols) {
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    if constexpr (L == Load::Vec4) {
        const float4* x4 = reinterpret_cast<const float4*>(x);
        const float4* b4 = reinterpret_cast<const float4*>(bias);
        float4* out4 = reinterpret_cast<float4*>(out);
        const size_t n4 = n / 4;
        const size_t cols4 = cols / 4;
        for (size_t i = tid; i < n4; i += stride) {
            const float4 a = x4[i];
            const float4 b = b4[i % cols4]; // bias
            out4[i] = {op(a.x, b.x), op(a.y, b.y), op(a.z, b.z), op(a.w, b.w)};
        }
    } else {
        for (size_t i = tid; i < n; i += stride) {
            out[i] = op(x[i], bias[i % cols]);
        }
    }
}

// ---- Host helpers ----

// Launch the Load instantiation picked at run time
template <typename Op, int N>
void launch_elementwise(Load l, Op op, Inputs<N> in, float* out, size_t n, cudaStream_t stream) {
    const unsigned grid = grid_size(work_items(n, l), K_BLOCK);
    if (l == Load::Vec4) {
        elementwise_kernel<Load::Vec4><<<grid, K_BLOCK, 0, stream>>>(op, in, out, n);
    } else {
        elementwise_kernel<Load::Scalar><<<grid, K_BLOCK, 0, stream>>>(op, in, out, n);
    }
    CUDL_KERNEL_CHECK();
}

template <typename Op>
void launch_bias(Load l, const float* x, const float* b, float* out, size_t n, size_t cols, cudaStream_t stream) {
    const unsigned grid = grid_size(work_items(n, l), K_BLOCK);
    if (l == Load::Vec4) {
        bias_kernel<Load::Vec4><<<grid, K_BLOCK, 0, stream>>>(Op{}, x, b, out, n, cols);
    } else {
        bias_kernel<Load::Scalar><<<grid, K_BLOCK, 0, stream>>>(Op{}, x, b, out, n, cols);
    }
    CUDL_KERNEL_CHECK();
}

// Host template executor shared by unary/binary/scale
template <Load L, typename Op, int N>
void elementwise(Op op, Inputs<N> in, float* out, size_t n, cudaStream_t stream) {
    if (n == 0)
        return;
    // use vec4 kernel only if data is 16byte aligned
    bool vec4 = (L == Load::Vec4 && aligned16(out));
    for (const float* p : in.p) {
        vec4 = vec4 && aligned16(p);
    }
    launch_elementwise(vec4 ? Load::Vec4 : Load::Scalar, op, in, out, n, stream);
}

// ---- Launchers (registry signatures) ----

// Unary
template <Load L, typename Op>
void unary(const float* in, float* out, size_t n, cudaStream_t stream) {
    elementwise<L>(Op{}, Inputs<1>{{in}}, out, n, stream);
}

// Scale doesnt need an Op
template <Load L>
void scale(const float* in, float alpha, float* out, size_t n, cudaStream_t stream) {
    elementwise<L>(ops::Scale{alpha}, Inputs<1>{{in}}, out, n, stream);
}

// Binary 
template <Load L, typename Op>
void binary(const float* a, const float* b, float* out, size_t n, cudaStream_t stream) {
    elementwise<L>(Op{}, Inputs<2>{{a, b}}, out, n, stream);
}

// Bias
template <Load L, typename Op>
void bias(const float* x, const float* b, float* out, size_t rows, size_t cols, cudaStream_t stream) {
    const size_t n = rows * cols;
    if (n == 0) return;
    // cols % 4 == 0 means n % 4 == 0, so no tail
    const bool vec4 = (L == Load::Vec4 && cols % 4 == 0 && aligned16(x) && aligned16(b) && aligned16(out));
    launch_bias<Op>(vec4 ? Load::Vec4 : Load::Scalar, x, b, out, n, cols, stream);
}

// ---- Registration ----

// Helper for adding kernels for both scalar and vec4 variants
template <typename Op>
void add_unary(const char* op) {
    add_loads(detail::raw_registry<UnaryFn>(), op, "", &unary<Load::Scalar, Op>, &unary<Load::Vec4, Op>);
}

template <typename Op>
void add_binary(const char* op) {
    add_loads(detail::raw_registry<BinaryFn>(), op, "", &binary<Load::Scalar, Op>, &binary<Load::Vec4, Op>);
}
}  // namespace


// Kernel registration
void register_elementwise_kernels() {
    // Unary
    add_unary<ops::Neg>("neg");
    add_unary<ops::Relu>("relu");
    add_unary<ops::Exp>("exp");
    add_unary<ops::Tanh>("tanh");
    add_unary<ops::Sigmoid>("sigmoid");

    // Binary
    add_binary<ops::Add>("add");
    add_binary<ops::Sub>("sub");
    add_binary<ops::Mul>("mul");
    add_binary<ops::Div>("div");
    add_binary<ops::ReluGrad>("relu_grad");

    // Scale
    add_loads(detail::raw_registry<ScaleFn>(), "scale", "", &scale<Load::Scalar>, &scale<Load::Vec4>);

    // Bias
    add_loads(detail::raw_registry<BiasFn>(), "add_bias", "", &bias<Load::Scalar, ops::Add>,
              &bias<Load::Vec4, ops::Add>);
}
}  // namespace cudl::kernels
