// Times every copy variant across sizes against cudaMemcpy (the practical ceiling)
// and checks the Phase 0 exit target: >80% of peak memory bandwidth.

#include <algorithm>
#include <cstdio>
#include <string>

#include "cudl/core/context.h"
#include "cudl/core/error.h"
#include "cudl/core/tensor.h"
#include "cudl/kernels/copy.h"
#include "cudl/kernels/registry.h"
#include "harness.h"

int main(int argc, char** argv) {
    using namespace cudl;

    bench::Harness harness(bench::Options::parse(argc, argv));
    const auto& dev = bench::DeviceInfo::get();
    std::printf("# %s, %d SMs, theoretical peak %.0f GB/s\n\n", dev.name.c_str(), dev.sm_count, dev.peak_bw_gbps);

    const int64_t sizes[] = {int64_t{1} << 20, int64_t{1} << 24, int64_t{1} << 26};  // 4 MB .. 256 MB
    double best_pct_largest = 0.0;

    for (const int64_t n : sizes) {
        Tensor src = Tensor::empty({n});
        Tensor dst = Tensor::empty({n});
        CUDA_CHECK(cudaMemsetAsync(src.data<float>(), 0, src.nbytes(), Context::get().stream()));

        const double bytes = 2.0 * static_cast<double>(src.nbytes());  // read + write
        const std::string shape = std::to_string(n);

        for (const auto& k : kernels::registry<kernels::CopyFn>().entries("copy")) {
            const auto r = harness.run("copy", k.variant, shape, bytes, 0.0, [&](cudaStream_t s) {
                k.fn(src.data<float>(), dst.data<float>(), static_cast<size_t>(n), s);
            });
            if (n == sizes[2]) best_pct_largest = std::max(best_pct_largest, r.pct_peak_bw());
        }

        // Practical ceiling to compare against.
        harness.run("copy", "cudaMemcpy", shape, bytes, 0.0, [&](cudaStream_t s) {
            CUDA_CHECK(
                cudaMemcpyAsync(dst.data<float>(), src.data<float>(), src.nbytes(), cudaMemcpyDeviceToDevice, s));
        });
    }

    std::printf("\nbest variant at largest size: %.1f%% of theoretical peak (Phase 0 target: >80%%)\n",
                best_pct_largest);
    return 0;
}
