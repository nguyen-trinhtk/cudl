#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace cudl::kernels {


// Reduction ops: "sum" and "max"
// Empty sum = 0
// Max: follows fmaxf (NaNs ignored, empty gives -inf)

// Max 1024 temp intermediate results
inline constexpr size_t kReduceWorkspaceFloats = 1024;

// Reduce all n elements into out[0]

// NOTE: `workspace`: mem for intermediate work
// must support at least kReduceWorkspaceFloats

// Variants:
//   atomic: per-thread partials combined with one atomic per thread 
//      NOTE: non-deterministic sum 
//      since atomic depends on memory ordering
//      and fp ops are not strictly associative
//   smem: per-block shared-memory tree, then a second single-block pass over the block partials
//   warp: warp shuffling to keep data in per-thread register
//   *_vec4: same as above with float4 loads in the per-thread partials
//      NOTE: falls back to scalar loads if `in` is not 16-byte aligned
using ReduceAllFn = void (*)(const float* in, float* out, size_t n, float* workspace, cudaStream_t stream);

// Row-wise reduction for a row-major matrix
// out[r] = reduce(in[r, :])
// Variants:
//   thread: one thread per row (serial, uncoalesced)
//   smem:   one block per row (shared-memory tree)
//   warp:   one warp per row (warp shuffles) 
// NOTE: Use warp for short rows and smem for longer rows
using ReduceRowsFn = void (*)(const float* in, float* out, size_t rows, size_t cols, cudaStream_t stream);

// Column-wise reduction for a row-major matrix
// out[c] = reduce(in[:, c])
// e.g. bias gradient sums dy over the batch
// Variants:
//   thread: one thread per column (neighbouring threads read neighbouring columns, so loads coalesce)
// TODO: smem/warp kernels?
using ReduceColsFn = void (*)(const float* in, float* out, size_t rows, size_t cols, cudaStream_t stream);

// Kernel registration
inline constexpr const char* kReduceOps[] = {"sum", "max"};
void register_reduce_kernels();
}  // namespace cudl::kernels
