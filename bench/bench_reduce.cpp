// Times every reduction variant: over all elements across sizes, and over the last dimension
// across row lengths. Checks the Phase 1 exit target: the best reduction at about 70% of peak
// memory bandwidth.

#include <algorithm>
#include <cstdio>
#include <map>
#include <string>

#include "cudl/core/context.h"
#include "cudl/core/error.h"
#include "cudl/core/tensor.h"
#include "cudl/kernels/reduce.h"
#include "cudl/kernels/registry.h"
#include "harness.h"

int main(int argc, char** argv) {
    using namespace cudl;
    using namespace cudl::kernels;

    bench::Harness harness(bench::Options::parse(argc, argv));
    const auto& dev = bench::DeviceInfo::get();
    std::printf("# %s, %d SMs, theoretical peak %.0f GB/s\n\n", dev.name.c_str(), dev.sm_count, dev.peak_bw_gbps);

    const int64_t sizes[] = {int64_t{1} << 20, int64_t{1} << 24, int64_t{1} << 26};  // 4 MB .. 256 MB
    const int64_t largest = sizes[2];
    std::map<std::string, double> best_pct;  // "op" or "op_rows" -> best %peakBW at the largest size

    Tensor out = Tensor::empty({largest});  // big enough for one result per row
    Tensor workspace = Tensor::empty({static_cast<int64_t>(kReduceWorkspaceFloats)});

    for (const int64_t n : sizes) {
        Tensor in = Tensor::empty({n});
        CUDA_CHECK(cudaMemsetAsync(in.data<float>(), 0, in.nbytes(), Context::get().stream()));
        const float* pi = in.data<float>();
        const double nb = static_cast<double>(in.nbytes());

        for (const char* op : kReduceOps) {
            for (const auto& k : registry<ReduceAllFn>().entries(op)) {
                const auto r = harness.run(op, k, std::to_string(n), nb, static_cast<double>(n), [&](cudaStream_t s) {
                    k.fn(pi, out.data<float>(), static_cast<size_t>(n), workspace.data<float>(), s);
                });
                if (r && n == largest) best_pct[op] = std::max(best_pct[op], r->pct_peak_bw());
            }

            // Last-dimension reductions over the same data, short to long rows.
            for (const int64_t cols : {int64_t{32}, int64_t{1024}, int64_t{32768}}) {
                const int64_t rows = n / cols;
                const std::string shape = std::to_string(rows) + "x" + std::to_string(cols);
                const std::string name = std::string(op) + "_rows";
                for (const auto& k : registry<ReduceRowsFn>().entries(op)) {
                    const auto r = harness.run(
                        name, k, shape, nb + rows * sizeof(float), static_cast<double>(n), [&](cudaStream_t s) {
                            k.fn(pi, out.data<float>(), static_cast<size_t>(rows), static_cast<size_t>(cols), s);
                        });
                    if (r && n == largest) best_pct[name] = std::max(best_pct[name], r->pct_peak_bw());
                }
            }
        }
    }

    std::printf("\nbest variant per op at n=%lld, over all row lengths for *_rows (Phase 1 target: about 70%%):\n",
                static_cast<long long>(largest));
    for (const auto& [op, pct] : best_pct) std::printf("  %-10s %5.1f%%\n", op.c_str(), pct);
    return 0;
}
