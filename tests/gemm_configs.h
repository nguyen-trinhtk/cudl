#pragma once

// Extra tile configs of the templated GEMM kernels, compiled only into the tests so the registry (and the
// benchmarks) keep one config per variant. test_gemm.cpp runs them through the same cases as registered variants.

#include <string>
#include <vector>

#include "cudl/kernels/gemm.h"

namespace cudl::test {

struct GemmConfig {
    std::string name;  // <variant>_<block tile>_<thread tile>, e.g. smem_2d_vec4_128x128x8_8x8
    kernels::GemmFn fn;
};

const std::vector<GemmConfig>& gemm_configs();

}  // namespace cudl::test
