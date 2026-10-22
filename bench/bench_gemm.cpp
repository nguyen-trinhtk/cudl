// Times every GEMM variant on square fp32 matrices; GFLOP/s is the number to watch.

#include <cstdio>
#include <string>

#include "cudl/core/context.h"
#include "cudl/core/error.h"
#include "cudl/core/tensor.h"
#include "cudl/kernels/gemm.h"
#include "cudl/kernels/registry.h"
#include "harness.h"

int main(int argc, char** argv) {
    using namespace cudl;
    using namespace cudl::kernels;

    bench::Harness harness(bench::Options::parse(argc, argv));
    const auto& dev = bench::DeviceInfo::get();
    std::printf("# %s, %d SMs, theoretical peak %.0f GB/s\n\n", dev.name.c_str(), dev.sm_count, dev.peak_bw_gbps);

    const int64_t sizes[] = {256, 512, 1024, 2048};

    for (const int64_t s : sizes) {
        Tensor a = Tensor::empty({s, s}), b = Tensor::empty({s, s}), c = Tensor::empty({s, s});
        const cudaStream_t stream = Context::get().stream();
        CUDA_CHECK(cudaMemsetAsync(a.data<float>(), 0, a.nbytes(), stream));
        CUDA_CHECK(cudaMemsetAsync(b.data<float>(), 0, b.nbytes(), stream));

        const float* pa = a.data<float>();
        const float* pb = b.data<float>();
        float* pc = c.data<float>();
        const size_t us = static_cast<size_t>(s);
        const std::string shape = std::to_string(s) + "x" + std::to_string(s) + "x" + std::to_string(s);
        // Minimum traffic: read A and B once, write C once.
        const double bytes = 3.0 * static_cast<double>(a.nbytes());
        const double flops = 2.0 * static_cast<double>(s) * s * s;

        for (const auto& k : registry<GemmFn>().entries("gemm")) {
            auto launch = [&](cudaStream_t st) {
                k.fn(pa, pb, pc, us, us, us, /*alpha=*/1.0f, /*beta=*/0.0f, st);
            };
            harness.run("gemm", k, shape, bytes, flops, launch);
        }
    }
    return 0;
}
