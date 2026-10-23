#include "cudl/kernels/gemm.h"
#include "cudl/kernels/registry.h"
#include "launch_utils.h"

// Kernels
#include "gemm/naive.cuh"
#include "gemm/smem_blocking.cuh"
#include "gemm/smem_1d_tiling.cuh"


// Kernels stay in gemm/*.cuh
// this file only registers them
namespace cudl::kernels {

// Kernel registration
void register_gemm_kernels() {
    detail::raw_registry<GemmFn>().add({"gemm", "naive", load_name(Load::Scalar), &gemm_impl::gemm_naive});
    detail::raw_registry<GemmFn>().add({"gemm", "smem", load_name(Load::Scalar), &gemm_impl::gemm_smem_blocking});
    detail::raw_registry<GemmFn>().add({"gemm", "smem_1d", load_name(Load::Scalar), &gemm_impl::gemm_smem_1d_tiling});
}
}  // namespace cudl::kernels
