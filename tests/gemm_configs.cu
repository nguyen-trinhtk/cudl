#include "gemm_configs.h"

#include "gemm/smem_1d_tiling.cuh"
#include "gemm/smem_2d_tiling.cuh"
#include "gemm/smem_blocking.cuh"

namespace cudl::test {

using kernels::Load;
using namespace kernels::gemm_impl;

// Common configs only; the registered defaults (src/kernels/gemm.cu) are left out since they're already tested
// Template args: smem <TILE_SIZE>, smem_1d <BM, BN, BK, TM>, smem_2d <BM, BN, BK, TM, TN, Load>
const std::vector<GemmConfig>& gemm_configs() {
    static const std::vector<GemmConfig> configs{
        {"smem_16", &gemm_smem_blocking<16>},
        {"smem_1d_64x64x16_8", &gemm_smem_1d_tiling<64, 64, 16, 8>},
        {"smem_1d_128x64x8_16", &gemm_smem_1d_tiling<128, 64, 8, 16>},
        {"smem_1d_32x32x32_4", &gemm_smem_1d_tiling<32, 32, 32, 4>},
        {"smem_2d_128x128x8_8x8", &gemm_smem_2d_tiling<128, 128, 8, 8, 8, Load::Scalar>},
        {"smem_2d_vec4_128x128x8_8x8", &gemm_smem_2d_tiling<128, 128, 8, 8, 8, Load::Vec4>},
        {"smem_2d_128x128x16_8x8", &gemm_smem_2d_tiling<128, 128, 16, 8, 8, Load::Scalar>},
        {"smem_2d_vec4_128x128x16_8x8", &gemm_smem_2d_tiling<128, 128, 16, 8, 8, Load::Vec4>},
        {"smem_2d_64x64x16_4x4", &gemm_smem_2d_tiling<64, 64, 16, 4, 4, Load::Scalar>},
        {"smem_2d_vec4_64x64x16_4x4", &gemm_smem_2d_tiling<64, 64, 16, 4, 4, Load::Vec4>},
    };
    return configs;
}

}  // namespace cudl::test
