#include "cudl/core/context.h"

#include "cudl/core/error.h"

namespace cudl {

Context& Context::get() {
    static Context ctx;
    return ctx;
}

Context::Context() {
    CUDA_CHECK(cudaGetDevice(&device_));
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_count_, cudaDevAttrMultiProcessorCount, device_));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
}

// Errors are ignored: this runs during static destruction.
Context::~Context() { cudaStreamDestroy(stream_); }

void Context::synchronize() const { CUDA_CHECK(cudaStreamSynchronize(stream_)); }

}  // namespace cudl
