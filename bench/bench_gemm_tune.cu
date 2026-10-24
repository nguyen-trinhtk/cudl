// Times the smem_2d vec4 GEMM for every tile size config (BM, BN, BK, TM, TN) on square fp32 matrices.
// Each config is first checked against the naive kernel, then timed. Prints a ranking by GFLOP/s at the end.
// --algo picks configs by name, e.g. --algo 'smem_2d_128x128x.*'; --size defaults to 4096.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "cudl/core/context.h"
#include "cudl/core/error.h"
#include "cudl/core/tensor.h"
#include "cudl/kernels/gemm.h"
#include "cudl/kernels/registry.h"
#include "gemm/smem_2d_tiling.cuh"
#include "harness.h"

namespace {

using cudl::kernels::GemmFn;
using cudl::kernels::Load;

// Tile sizes to try
constexpr int kBM[] = {64, 128, 256};
constexpr int kBN[] = {64, 128, 256};
constexpr int kBK[] = {8, 16, 32};
constexpr int kTM[] = {4, 8};
constexpr int kTN[] = {4, 8};
constexpr size_t kNumBM = std::size(kBM), kNumBN = std::size(kBN), kNumBK = std::size(kBK), kNumTM = std::size(kTM),
                 kNumTN = std::size(kTN);
constexpr size_t kGridSize = kNumBM * kNumBN * kNumBK * kNumTM * kNumTN;
constexpr int kMinThreads = 64;

struct Config {
    int bm, bn, bk, tm, tn;
    GemmFn fn;

    int threads() const { return (bm / tm) * (bn / tn); }
    std::string name() const {
        return "smem_2d_" + std::to_string(bm) + "x" + std::to_string(bn) + "x" + std::to_string(bk) + "_" +
               std::to_string(tm) + "x" + std::to_string(tn);
    }
};

// Same checks as the static_asserts in gemm_smem_2d_tiling: configs that fail are skipped
template <int BM, int BN, int BK, int TM, int TN>
constexpr bool valid() {
    constexpr int threads = (BM / TM) * (BN / TN);
    return BM % TM == 0 && BN % TN == 0 && threads >= kMinThreads && threads <= 1024 &&
           (BM * BK + BK * BN) * sizeof(float) <= 48 * 1024 && (BM * BK / 4) % threads == 0 &&
           (BK * BN / 4) % threads == 0;
}

// Config number I out of all (BM, BN, BK, TM, TN) combinations
template <size_t I>
void add_config(std::vector<Config>& out) {
    constexpr int BM = kBM[I / (kNumBN * kNumBK * kNumTM * kNumTN)];
    constexpr int BN = kBN[I / (kNumBK * kNumTM * kNumTN) % kNumBN];
    constexpr int BK = kBK[I / (kNumTM * kNumTN) % kNumBK];
    constexpr int TM = kTM[I / kNumTN % kNumTM];
    constexpr int TN = kTN[I % kNumTN];
    if constexpr (valid<BM, BN, BK, TM, TN>()) {
        out.push_back({BM, BN, BK, TM, TN, &cudl::kernels::gemm_impl::gemm_smem_2d_tiling<BM, BN, BK, TM, TN, Load::Vec4>});
    }
}

template <size_t... Is>
std::vector<Config> all_configs(std::index_sequence<Is...>) {
    std::vector<Config> out;
    (add_config<Is>(out), ...);
    return out;
}

std::vector<float> random_matrix(size_t n, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> out(n);
    for (float& x : out) x = dist(gen);
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    using namespace cudl;
    using namespace cudl::kernels;

    const bench::Options opts = bench::Options::parse(argc, argv);
    bench::Harness harness(opts);
    const auto& dev = bench::DeviceInfo::get();
    std::printf("# %s, %d SMs\n", dev.name.c_str(), dev.sm_count);

    const std::vector<Config> configs = all_configs(std::make_index_sequence<kGridSize>{});
    std::printf("# %zu of %zu grid points are valid configs\n\n", configs.size(), kGridSize);

    const std::vector<int64_t> sizes = opts.sizes.empty() ? std::vector<int64_t>{4096} : opts.sizes;
    const cudaStream_t stream = Context::get().stream();

    for (const int64_t s : sizes) {
        const size_t us = static_cast<size_t>(s);
        const size_t numel = us * us;
        Tensor a = Tensor::from_vector(random_matrix(numel, 1), {s, s});
        Tensor b = Tensor::from_vector(random_matrix(numel, 2), {s, s});
        Tensor c = Tensor::empty({s, s}), c_ref = Tensor::empty({s, s});
        const float* pa = a.data<float>();
        const float* pb = b.data<float>();
        float* pc = c.data<float>();

        const std::string shape = std::to_string(s) + "x" + std::to_string(s) + "x" + std::to_string(s);
        const double bytes = 3.0 * static_cast<double>(a.nbytes());
        const double flops = 2.0 * static_cast<double>(s) * s * s;

        // Reference output from the naive kernel
        registry<GemmFn>().get("gemm", "naive").fn(pa, pb, c_ref.data<float>(), us, us, us, false, false, 1.0f, 0.0f,
                                                   stream);
        std::vector<float> ref(numel), out(numel);
        CUDA_CHECK(cudaMemcpyAsync(ref.data(), c_ref.data<float>(), c_ref.nbytes(), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::vector<std::pair<Config, bench::Result>> ranked;
        std::vector<std::string> failed;
        for (const Config& cfg : configs) {
            KernelEntry<GemmFn> k{"gemm", cfg.name(), kLoadVec4, cfg.fn};
            k.variant = variant_name(k.algo, k.load);
            if (!harness.selected("gemm", k.algo, k.load)) continue;

            auto launch = [&](cudaStream_t st) {
                k.fn(pa, pb, pc, us, us, us, /*trans_a=*/false, /*trans_b=*/false, /*alpha=*/1.0f, /*beta=*/0.0f, st);
            };

            // Check against naive before timing (allow small float rounding differences)
            CUDA_CHECK(cudaMemsetAsync(pc, 0xff, c.nbytes(), stream));  // fill with NaN so missed outputs fail
            launch(stream);
            CUDA_CHECK(cudaMemcpyAsync(out.data(), pc, c.nbytes(), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            const float tol = 1e-5f * static_cast<float>(s);
            size_t bad = 0;
            for (size_t i = 0; i < numel; ++i)
                if (!(std::fabs(out[i] - ref[i]) <= tol * (1.0f + std::fabs(ref[i])))) ++bad;
            if (bad) {
                std::printf("# MISMATCH %s: %zu of %zu outputs off\n", k.algo.c_str(), bad, numel);
                failed.push_back(k.algo);
                continue;
            }

            if (auto r = harness.run("gemm", k, shape, bytes, flops, launch)) ranked.emplace_back(cfg, *r);
        }

        std::sort(ranked.begin(), ranked.end(),
                  [](const auto& x, const auto& y) { return x.second.gflops() > y.second.gflops(); });
        std::printf("\n# Ranking for %s (by GFLOP/s)\n", shape.c_str());
        std::printf("%4s %5s %5s %5s %4s %4s %7s %10s %9s\n", "rank", "BM", "BN", "BK", "TM", "TN", "threads",
                    "median_ms", "GFLOP/s");
        for (size_t i = 0; i < ranked.size(); ++i) {
            const auto& [cfg, r] = ranked[i];
            std::printf("%4zu %5d %5d %5d %4d %4d %7d %10.4f %9.1f\n", i + 1, cfg.bm, cfg.bn, cfg.bk, cfg.tm, cfg.tn,
                        cfg.threads(), r.median_ms, r.gflops());
        }
        if (!failed.empty()) std::printf("# %zu configs failed the correctness check\n", failed.size());
        std::printf("\n");
    }
    return 0;
}
