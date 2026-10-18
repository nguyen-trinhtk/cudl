// Times every elementwise op and variant across sizes against cudaMemcpy (the practical ceiling)
// and checks the Phase 1 exit target: elementwise kernels at about 80% of peak memory bandwidth.

#include <algorithm>
#include <cstdio>
#include <map>
#include <optional>
#include <string>

#include "cudl/core/context.h"
#include "cudl/core/error.h"
#include "cudl/core/tensor.h"
#include "cudl/kernels/elementwise.h"
#include "cudl/kernels/registry.h"
#include "harness.h"

int main(int argc, char** argv) {
    using namespace cudl;
    using namespace cudl::kernels;

    bench::Harness harness(bench::Options::parse(argc, argv));
    const auto& dev = bench::DeviceInfo::get();
    std::printf("# %s, %d SMs, theoretical peak %.0f GB/s\n\n", dev.name.c_str(), dev.sm_count, dev.peak_bw_gbps);

    const int64_t sizes[] = {int64_t{1} << 20, int64_t{1} << 24, int64_t{1} << 26};  // 4 MB .. 256 MB per tensor
    const int64_t largest = sizes[2];
    constexpr int64_t kBiasCols = 1024;
    std::map<std::string, double> best_pct;  // op -> best variant's %peakBW at the largest size

    for (const int64_t n : sizes) {
        Tensor a = Tensor::empty({n}), b = Tensor::empty({n}), out = Tensor::empty({n});
        const cudaStream_t stream = Context::get().stream();
        CUDA_CHECK(cudaMemsetAsync(a.data<float>(), 0, a.nbytes(), stream));
        // 0x3f3f3f3f is about 0.75f: a safe divisor.
        CUDA_CHECK(cudaMemsetAsync(b.data<float>(), 0x3f, b.nbytes(), stream));

        const float* pa = a.data<float>();
        const float* pb = b.data<float>();
        float* po = out.data<float>();
        const size_t un = static_cast<size_t>(n);
        const double nb = static_cast<double>(a.nbytes());
        const std::string shape = std::to_string(n);

        auto record = [&](const std::string& op, const std::optional<bench::Result>& r) {
            if (r && n == largest) best_pct[op] = std::max(best_pct[op], r->pct_peak_bw());
        };

        for (const char* op : kUnaryOps)
            for (const auto& k : registry<UnaryFn>().entries(op))
                record(op, harness.run(op, k, shape, 2 * nb, static_cast<double>(n),
                                       [&](cudaStream_t s) { k.fn(pa, po, un, s); }));

        for (const auto& k : registry<ScaleFn>().entries("scale"))
            record("scale", harness.run("scale", k, shape, 2 * nb, static_cast<double>(n),
                                        [&](cudaStream_t s) { k.fn(pa, 0.5f, po, un, s); }));

        for (const char* op : kBinaryOps)
            for (const auto& k : registry<BinaryFn>().entries(op))
                record(op, harness.run(op, k, shape, 3 * nb, static_cast<double>(n),
                                       [&](cudaStream_t s) { k.fn(pa, pb, po, un, s); }));

        const size_t rows = static_cast<size_t>(n / kBiasCols);
        const std::string bias_shape = std::to_string(rows) + "x" + std::to_string(kBiasCols);
        for (const auto& k : registry<BiasFn>().entries("add_bias"))
            record("add_bias",
                   harness.run("add_bias", k, bias_shape, 2 * nb + kBiasCols * sizeof(float),
                               static_cast<double>(n), [&](cudaStream_t s) { k.fn(pa, pb, po, rows, kBiasCols, s); }));

        // Practical ceiling to compare against (2 bytes moved per byte copied, like a unary op).
        harness.run_baseline("copy", "cudaMemcpy", shape, 2 * nb, 0.0, [&](cudaStream_t s) {
            CUDA_CHECK(cudaMemcpyAsync(po, pa, a.nbytes(), cudaMemcpyDeviceToDevice, s));
        });
    }

    std::printf("\nbest variant per op at n=%lld (Phase 1 target: about 80%% of theoretical peak):\n",
                static_cast<long long>(largest));
    for (const auto& [op, pct] : best_pct) std::printf("  %-10s %5.1f%%\n", op.c_str(), pct);
    return 0;
}
