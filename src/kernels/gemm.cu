#include "cudl/kernels/gemm.h"
#include "cudl/kernels/registry.h"
#include "launch_utils.h"

// Kernels
#include "gemm/naive.cuh"
#include "gemm/smem_blocking.cuh"
#include "gemm/smem_1d_tiling.cuh"
#include "gemm/smem_2d_tiling.cuh"


// Kernels stay in gemm/*.cuh
// this file only registers them
namespace cudl::kernels {

// Kernel registration
void register_gemm_kernels() {
    // Template args: naive <BX, BY>, smem <TILE_SIZE>, smem_1d <BM, BN, BK, TM>, smem_2d <BM, BN, BK, TM, TN>
    detail::raw_registry<GemmFn>().add({"gemm", "naive", load_name(Load::Scalar), &gemm_impl::gemm_naive<32, 32>});
    detail::raw_registry<GemmFn>().add({"gemm", "smem", load_name(Load::Scalar), &gemm_impl::gemm_smem_blocking<32>});
    detail::raw_registry<GemmFn>().add({"gemm", "smem_1d", load_name(Load::Scalar), &gemm_impl::gemm_smem_1d_tiling<64, 64, 8, 8>});
    detail::raw_registry<GemmFn>().add({"gemm", "smem_2d", load_name(Load::Scalar), &gemm_impl::gemm_smem_2d_tiling<64, 64, 8, 8, 8>});
}
}  // namespace cudl::kernels
