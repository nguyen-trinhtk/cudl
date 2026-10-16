#pragma once

#include <cuda_runtime.h>

namespace cudl {

// Process-wide execution state: current device and the stream all work goes on.
// Later phases add the cuBLAS handle (benchmark baseline) and RNG state here.
class Context {
public:
    static Context& get();

    cudaStream_t stream() const { return stream_; }
    int device() const { return device_; }
    int sm_count() const { return sm_count_; }
    void synchronize() const;

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

private:
    Context();
    ~Context();

    int device_ = 0;
    int sm_count_ = 0;
    cudaStream_t stream_ = nullptr;
};

}  // namespace cudl
