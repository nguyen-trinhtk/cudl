#include <algorithm>

#include "cudl/core/error.h"
#include "cudl/kernels/reduce.h"
#include "cudl/kernels/registry.h"
#include "launch_utils.h"
#include "reduce_ops.cuh"

namespace cudl::kernels {
namespace {

constexpr int K_BLOCK = K_DEFAULT_BLOCK;  // num threads per block (must be mult of K_WARP)
static_assert(K_BLOCK % K_WARP == 0 && K_BLOCK <= K_MAX_BLOCK, "warp shuffles need whole warps");

// Reduce-all variants: how per-thread partials are combined
enum class Variant { Atomic, Smem, Warp };

// ---- Device helpers ----

// Shared memory tree reduce
template <typename Op>
__device__ float block_reduce_smem(Op op, float v, float* smem) {
    smem[threadIdx.x] = v; // write to shared mem
    __syncthreads(); // sync after smem write
    // descending powers of 2 (e.g. 128, 64, 32, ..., 2, 1)
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) { // depth = log2(blockDim)
        // merge elements in first half with second half
        if (threadIdx.x < s) {
            smem[threadIdx.x] = op(smem[threadIdx.x], smem[threadIdx.x + s]);
        }
        __syncthreads(); // sync after smem write
    }
    const float out = smem[0]; // result will be in first element
    __syncthreads();  // final sync since smem may be reused
    return out;
}

// Warp shuffle reduce (utilize per-thread registers)
// Reduce 32 floats into 1 (each warp got 1 result)
// Will need to call block_reduce_warp to finalize
template <typename Op>
__device__ __forceinline__ float warp_reduce(Op op, float v) {
    // Offsets = desc powers of 2 (16, 8, 4, 2, 1)
    for (int offset = K_WARP / 2; offset > 0; offset >>= 1) {
        // cross-thread shuffle
        v = op(v, __shfl_down_sync(0xffffffffu, v, offset)); 
    }
    return v;
}

// Shuffle within each warp, then the first warp reduces the per-warp results. Valid in thread 0.
template <typename Op>
__device__ float block_reduce_warp(Op op, float v) {
    __shared__ float warp_vals[MAX_NUM_WARP];
    const unsigned lane = threadIdx.x % K_WARP; // lane within a warp
    const unsigned warp_id = threadIdx.x / K_WARP; // warp # within block
    v = warp_reduce(op, v); // reduce in-warp
    // first lane of warp holds warp-reduce result
    if (lane == 0) {
        warp_vals[warp_id] = v;
    }
    __syncthreads(); // after write smem

    // Final reduce in warp_vals
    const size_t num_warps = blockDim.x / K_WARP;
    if (warp_id == 0) {
        // Only first num_warps slots have actual value
        // The rest are empty (use identity())
        v = (lane < num_warps) ? warp_vals[lane] : Op::identity();
        v = warp_reduce(op, v);
    }
    __syncthreads();  // safety sync
    return v;
}

// Grid-stride partial reduction of in[0, n) for this thread.
template <typename Op>
__device__ float thread_partial(Op op, const float* __restrict__ in, size_t n) {
    float acc = Op::identity();
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride)
        acc = op(acc, in[i]);
    return acc;
}

// Same with float4 loads; `in` must be 16-byte aligned.
template <typename Op>
__device__ float thread_partial_vec4(Op op, const float* __restrict__ in, size_t n) {
    float acc = Op::identity();
    const size_t n4 = n / 4;
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = tid; i < n4; i += stride) {
        const float4 v = reinterpret_cast<const float4*>(in)[i];
        acc = op(acc, op(op(v.x, v.y), op(v.z, v.w)));
    }
    if (tid < n - n4 * 4) acc = op(acc, in[n4 * 4 + tid]);
    return acc;
}

// Per-thread partial, with float4 loads for Load::Vec4
template <Load L, typename Op>
__device__ __forceinline__ float load_partial(Op op, const float* __restrict__ in, size_t n) {
    if constexpr (L == Load::Vec4) {
        return thread_partial_vec4(op, in, n);
    } else {
        return thread_partial(op, in, n);
    }
}

// ---- Kernels ----

// Fill one kernel
__global__ void fill_one_kernel(float* out, float v) {
    *out = v;
}

// Atomic: every thread combines into out[0]
// Smem/Warp: one partial per block written to out[blockIdx.x]
template <Variant V, Load L, typename Op>
__global__ void reduce_all_kernel(Op op, const float* __restrict__ in, float* out, size_t n) {
    const float acc = load_partial<L>(op, in, n);
    if constexpr (V == Variant::Atomic) {
        Op::atomic(out, acc);
    } else {
        float v;
        if constexpr (V == Variant::Smem) {
            __shared__ float smem[K_BLOCK];
            v = block_reduce_smem(op, acc, smem);
        } else {
            v = block_reduce_warp(op, acc);
        }
        if (threadIdx.x == 0)
            out[blockIdx.x] = v;
    }
}

// One thread per row. Neighbouring threads read addresses `cols` apart, so loads are uncoalesced.
template <typename Op>
__global__ void reduce_rows_thread_kernel(Op op, const float* __restrict__ in, float* __restrict__ out, size_t rows,
                                          size_t cols) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t r = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; r < rows; r += stride) {
        const float* row = in + r * cols;
        float acc = Op::identity();
        for (size_t c = 0; c < cols; ++c) acc = op(acc, row[c]);
        out[r] = acc;
    }
}

// Shared mem: One block per row.
template <typename Op>
__global__ void reduce_rows_smem_kernel(Op op, const float* __restrict__ in, float* __restrict__ out, size_t rows,
                                        size_t cols) {
    __shared__ float smem[K_BLOCK];
    for (size_t r = blockIdx.x; r < rows; r += gridDim.x) {
        const float* row = in + r * cols;
        float acc = Op::identity();
        for (size_t c = threadIdx.x; c < cols; c += blockDim.x) acc = op(acc, row[c]);
        acc = block_reduce_smem(op, acc, smem);
        if (threadIdx.x == 0) 
            out[r] = acc;
    }
}

// One warp per row
// The whole warp shares `r`, so full-mask shuffles are safe.
template <typename Op>
__global__ void reduce_rows_warp_kernel(Op op, const float* __restrict__ in, float* __restrict__ out, size_t rows,
                                        size_t cols) {
    const unsigned lane = threadIdx.x % K_WARP;
    const size_t num_warp = static_cast<size_t>(gridDim.x) * blockDim.x / K_WARP;
    for (size_t r = (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) / K_WARP; r < rows; r += num_warp) {
        const float* row = in + r * cols;
        float acc = Op::identity();
        for (size_t c = lane; c < cols; c += K_WARP) {
            acc = op(acc, row[c]);
        }
        acc = warp_reduce(op, acc);
        if (lane == 0) out[r] = acc;
    }
}

// One thread per column, already coalesced
template <typename Op>
__global__ void reduce_cols_thread_kernel(Op op, const float* __restrict__ in, float* __restrict__ out, size_t rows,
                                          size_t cols) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t c = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; c < cols; c += stride) {
        float acc = Op::identity();
        for (size_t r = 0; r < rows; ++r) {
            acc = op(acc, in[r * cols + c]);
        }
        out[c] = acc;
    }
}

// ---- Host helpers ----

// Block count for two-pass reductions: one partial per block must fit in the workspace.
unsigned partial_blocks(size_t items) {
    return std::min<unsigned>(grid_size(items, K_BLOCK), static_cast<unsigned>(kReduceWorkspaceFloats));
}

// Load actually used for `in`: Vec4 only if requested and data is 16byte aligned
template <Load L>
Load resolve_load(const float* in) {
    return (L == Load::Vec4 && aligned16(in)) ? Load::Vec4 : Load::Scalar;
}

// Launch the Load instantiation picked at run time
template <Variant V, typename Op>
void launch_reduce_all(Load l, unsigned grid, const float* in, float* out, size_t n, cudaStream_t stream) {
    if (l == Load::Vec4) {
        reduce_all_kernel<V, Load::Vec4><<<grid, K_BLOCK, 0, stream>>>(Op{}, in, out, n);
    } else {
        reduce_all_kernel<V, Load::Scalar><<<grid, K_BLOCK, 0, stream>>>(Op{}, in, out, n);
    }
    CUDL_KERNEL_CHECK();
}

// ---- Launchers (registry signatures) ----

// Reduce all
// Atomic: single pass straight into out
// Smem/Warp: pass 1 writes per-block partials to the workspace, pass 2 reduces them with a single block
template <Variant V, Load L, typename Op>
void reduce_all(const float* in, float* out, size_t n, float* workspace, cudaStream_t stream) {
    const Load l = resolve_load<L>(in);
    if constexpr (V == Variant::Atomic) {
        fill_one_kernel<<<1, 1, 0, stream>>>(out, Op::identity());
        CUDL_KERNEL_CHECK();
        if (n == 0)
            return;
        launch_reduce_all<V, Op>(l, grid_size(work_items(n, l), K_BLOCK), in, out, n, stream);
    } else {
        const unsigned blocks = partial_blocks(work_items(n, l));
        if (blocks == 1) {
            launch_reduce_all<V, Op>(l, 1, in, out, n, stream);
            return;
        }
        // P1: Partial per-block results
        launch_reduce_all<V, Op>(l, blocks, in, workspace, n, stream);
        // P2: Full grid reduced
        launch_reduce_all<V, Op>(resolve_load<L>(workspace), 1, workspace, out, blocks, stream);
    }
}

// Reduce rows: one launcher per variant, each with its own grid shape
template <typename Op>
void reduce_rows_thread(const float* in, float* out, size_t rows, size_t cols, cudaStream_t stream) {
    if (rows == 0) 
        return;
    reduce_rows_thread_kernel<<<grid_size(rows, K_BLOCK), K_BLOCK, 0, stream>>>(Op{}, in, out, rows, cols);
    CUDL_KERNEL_CHECK();
}

template <typename Op>
void reduce_rows_smem(const float* in, float* out, size_t rows, size_t cols, cudaStream_t stream) {
    if (rows == 0) 
        return;
    reduce_rows_smem_kernel<<<grid_size(rows, 1), K_BLOCK, 0, stream>>>(Op{}, in, out, rows, cols);
    CUDL_KERNEL_CHECK();
}

template <typename Op>
void reduce_rows_warp(const float* in, float* out, size_t rows, size_t cols, cudaStream_t stream) {
    if (rows == 0) 
        return;
    reduce_rows_warp_kernel<<<grid_size(rows * K_WARP, K_BLOCK), K_BLOCK, 0, stream>>>(Op{}, in, out, rows, cols);
    CUDL_KERNEL_CHECK();
}

template <typename Op>
void reduce_cols_thread(const float* in, float* out, size_t rows, size_t cols, cudaStream_t stream) {
    if (cols == 0) return;
    reduce_cols_thread_kernel<<<grid_size(cols, K_BLOCK), K_BLOCK, 0, stream>>>(Op{}, in, out, rows, cols);
    CUDL_KERNEL_CHECK();
}

// ---- Registration ----

// Helper to add reduce kernels
template <typename Op>
void add_reduce(const char* op, bool atomic_deterministic) {
    // Reduce all
    auto& ra = detail::raw_registry<ReduceAllFn>();
    add_loads(ra, op, "atomic", &reduce_all<Variant::Atomic, Load::Scalar, Op>,
              &reduce_all<Variant::Atomic, Load::Vec4, Op>, atomic_deterministic);
    add_loads(ra, op, "smem", &reduce_all<Variant::Smem, Load::Scalar, Op>, &reduce_all<Variant::Smem, Load::Vec4, Op>);
    add_loads(ra, op, "warp", &reduce_all<Variant::Warp, Load::Scalar, Op>, &reduce_all<Variant::Warp, Load::Vec4, Op>);

    // Reduce rows (scalar loads only)
    auto& rr = detail::raw_registry<ReduceRowsFn>();
    rr.add({op, "thread", kLoadScalar, &reduce_rows_thread<Op>});
    rr.add({op, "smem", kLoadScalar, &reduce_rows_smem<Op>});
    rr.add({op, "warp", kLoadScalar, &reduce_rows_warp<Op>});

    // Reduce cols (scalar loads only)
    detail::raw_registry<ReduceColsFn, ReduceColsTag>().add({op, "thread", kLoadScalar, &reduce_cols_thread<Op>});
}
}  // namespace


// Reduce kernels
void register_reduce_kernels() {
    add_reduce<SumOp>("sum", /*atomic_deterministic=*/false); // non-deterministic atomic variant
    add_reduce<MaxOp>("max", /*atomic_deterministic=*/true); // deterministic atomic variant
}
}  // namespace cudl::kernels
