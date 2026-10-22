#include "cudl/kernels/gemm.h"
#include "cudl/kernels/registry.h"
#include "gemm/naive.cuh"
#include "launch_utils.h"

// Kernels stay in gemm/*.cuh
// this file only registers them
namespace cudl::kernels {

// Kernel registration
void register_gemm_kernels() {
    detail::raw_registry<GemmFn>().add({"gemm", "naive", load_name(Load::Scalar), &gemm_impl::gemm_naive});
}
}  // namespace cudl::kernels
