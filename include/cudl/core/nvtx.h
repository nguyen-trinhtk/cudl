#pragma once

#ifndef CUDL_NO_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

namespace cudl {

// Scoped NVTX range; shows up as a named span in Nsight Systems timelines.
class NvtxRange {
public:
    explicit NvtxRange(const char* name) {
#ifndef CUDL_NO_NVTX
        nvtxRangePushA(name);
#else
        (void)name;
#endif
    }
    ~NvtxRange() {
#ifndef CUDL_NO_NVTX
        nvtxRangePop();
#endif
    }
    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
};

}  // namespace cudl

#define CUDL_NVTX_CONCAT_(a, b) a##b
#define CUDL_NVTX_CONCAT(a, b) CUDL_NVTX_CONCAT_(a, b)
#define CUDL_NVTX_RANGE(name) ::cudl::NvtxRange CUDL_NVTX_CONCAT(cudl_nvtx_range_, __LINE__)(name)
