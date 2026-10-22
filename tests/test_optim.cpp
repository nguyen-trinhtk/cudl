// Optimizer kernels: several steps of SGD (with and without momentum) and Adam on random grads,
// checked against the double CPU reference after every step, plus guard checks.

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <tuple>
#include <vector>

#include "common/compare.h"
#include "common/guarded.h"
#include "common/random.h"
#include "cudl/core/context.h"
#include "cudl/kernels/optim.h"
#include "cudl/kernels/registry.h"
#include "ref/optim_ref.h"

namespace cudl {
namespace {

using kernels::AdamFn;
using kernels::SgdFn;

constexpr int kSteps = 5;
// Each fp32 step rounds once or twice; the error compounds over kSteps.
constexpr test::Tolerance kOptimTol{1e-6, 1e-5, -1};

const std::vector<size_t> kSizes{0, 1, 5, 257, 4099, (1 << 16) + 3};

std::vector<double> to_double(const std::vector<float>& v) {
    return {v.begin(), v.end()};
}

// ---------------------------------------------------------------------------- SGD

using SgdParam = std::tuple<std::string, size_t, float>;  // (variant, n, momentum)

class SgdKernelTest : public ::testing::TestWithParam<SgdParam> {};

std::string SgdName(const ::testing::TestParamInfo<SgdParam>& info) {
    const auto& [variant, n, momentum] = info.param;
    return variant + "_n" + std::to_string(n) + (momentum == 0.0f ? "_plain" : "_momentum");
}

TEST_P(SgdKernelTest, MatchesReferenceEveryStep) {
    const auto& [variant, n, momentum] = GetParam();
    const auto& k = kernels::registry<SgdFn>().get("sgd", variant);
    const float lr = 0.1f;
    const auto host_p = test::random_uniform(n, /*seed=*/n + 1);

    test::Guarded param(host_p, 1), velocity(std::vector<float>(n, 0.0f), 1);
    std::vector<double> ref_p = to_double(host_p), ref_v(n, 0.0);
    for (int step = 0; step < kSteps; ++step) {
        const auto host_g = test::random_uniform(n, /*seed=*/n * 7 + step);
        test::Guarded grad(host_g, 1);
        k.fn(param.data(), grad.data(), momentum == 0.0f ? nullptr : velocity.data(), n, lr, momentum,
             Context::get().stream());
        Context::get().synchronize();
        ref::sgd_step(ref_p, to_double(host_g), ref_v, lr, momentum);
        EXPECT_TRUE(test::AllClose(param.body(), ref_p, kOptimTol)) << "step " << step;
    }
    EXPECT_TRUE(param.guards_intact());
    EXPECT_TRUE(velocity.guards_intact());
}

INSTANTIATE_TEST_SUITE_P(AllVariants, SgdKernelTest,
                         ::testing::Combine(::testing::ValuesIn(kernels::registry<SgdFn>().variants("sgd")),
                                            ::testing::ValuesIn(kSizes), ::testing::Values(0.0f, 0.9f)),
                         SgdName);

// ---------------------------------------------------------------------------- Adam

using AdamParam = std::tuple<std::string, size_t>;  // (variant, n)

class AdamKernelTest : public ::testing::TestWithParam<AdamParam> {};

std::string AdamName(const ::testing::TestParamInfo<AdamParam>& info) {
    const auto& [variant, n] = info.param;
    return variant + "_n" + std::to_string(n);
}

TEST_P(AdamKernelTest, MatchesReferenceEveryStep) {
    const auto& [variant, n] = GetParam();
    const auto& k = kernels::registry<AdamFn>().get("adam", variant);
    const float lr = 1e-2f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    const auto host_p = test::random_uniform(n, /*seed=*/n + 2);

    test::Guarded param(host_p, 1), m(std::vector<float>(n, 0.0f), 1), v(std::vector<float>(n, 0.0f), 1);
    std::vector<double> ref_p = to_double(host_p), ref_m(n, 0.0), ref_v(n, 0.0);
    for (int step = 1; step <= kSteps; ++step) {
        const auto host_g = test::random_uniform(n, /*seed=*/n * 11 + step);
        test::Guarded grad(host_g, 1);
        const double bc1 = 1.0 - std::pow(static_cast<double>(beta1), step);
        const double bc2 = 1.0 - std::pow(static_cast<double>(beta2), step);
        const kernels::AdamStep hp{lr, beta1, beta2, eps, static_cast<float>(bc1), static_cast<float>(bc2)};
        k.fn(param.data(), grad.data(), m.data(), v.data(), n, hp, Context::get().stream());
        Context::get().synchronize();
        ref::adam_step(ref_p, to_double(host_g), ref_m, ref_v, step, lr, beta1, beta2, eps);
        EXPECT_TRUE(test::AllClose(param.body(), ref_p, kOptimTol)) << "step " << step;
    }
    EXPECT_TRUE(param.guards_intact());
    EXPECT_TRUE(m.guards_intact());
    EXPECT_TRUE(v.guards_intact());
}

INSTANTIATE_TEST_SUITE_P(AllVariants, AdamKernelTest,
                         ::testing::Combine(::testing::ValuesIn(kernels::registry<AdamFn>().variants("adam")),
                                            ::testing::ValuesIn(kSizes)),
                         AdamName);

}  // namespace
}  // namespace cudl
