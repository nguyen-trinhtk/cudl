// GEMM kernels: every registered variant and extra tile config (gemm_configs.cu) x shape x transposes x (alpha, beta)
// x alignment offset, checked against the CPU reference plus guard checks. Also: beta == 0 must not read C, one
// misaligned operand must not break the vec4 path, and long-K accumulation error stays under the fp32 dot-product bound.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "common/compare.h"
#include "common/guarded.h"
#include "common/random.h"
#include "cudl/core/context.h"
#include "cudl/kernels/gemm.h"
#include "cudl/kernels/registry.h"
#include "gemm_configs.h"
#include "ref/gemm_ref.h"

namespace cudl {
namespace {

using kernels::GemmFn;

struct Shape {
    size_t m, n, k;
};

struct Scalars {
    float alpha, beta;
};

std::vector<std::string> config_names() {
    std::vector<std::string> names;
    for (const auto& cfg : test::gemm_configs()) {
        names.push_back(cfg.name);
    }
    return names;
}

// Registered variants, then the extra tile configs
std::vector<std::string> kernel_names() {
    auto names = kernels::registry<GemmFn>().variants("gemm");
    for (const auto& name : config_names()) {
        names.push_back(name);
    }
    return names;
}

GemmFn kernel_fn(const std::string& name) {
    for (const auto& cfg : test::gemm_configs()) {
        if (cfg.name == name) {
            return cfg.fn;
        }
    }
    return kernels::registry<GemmFn>().get("gemm", name).fn;
}

// Includes empty dims, K == 0, and shapes that don't divide the tile
const std::vector<Shape> kShapes{{0, 4, 4},   {4, 0, 4},     {3, 5, 0},      {1, 1, 1},       {1, 7, 13},
                                 {16, 16, 16}, {17, 33, 9},  {127, 65, 33},  {64, 256, 128},  {300, 200, 513}};
const std::vector<Scalars> kScalars{{1.0f, 0.0f}, {1.0f, 1.0f}, {0.5f, -2.0f}, {0.0f, 0.75f}};
const std::vector<size_t> kOffsets{0, 1};
const std::vector<std::pair<bool, bool>> kTrans{{false, false}, {true, false}, {false, true}, {true, true}};

// Pass iff |err| <= gamma_{k+2} * scale for every element, scale = |alpha||A||B| + |beta||C|: the worst-case fp32
// rounding bound for any summation order (see ref::gemm_error_scale), so no atol / magic numbers are needed
test::Tolerance gemm_tolerance(size_t k) {
    return {0.0, ref::gamma_f32(k + 2), -1};
}

// Checks a GEMM result against the reference under the rounding bound above
::testing::AssertionResult GemmClose(const std::vector<float>& actual, const std::vector<float>& a,
                                     const std::vector<float>& b, const std::vector<float>& c, size_t m, size_t n,
                                     size_t k, bool trans_a, bool trans_b, float alpha, float beta) {
    const auto expected = ref::gemm(a, b, c, m, n, k, trans_a, trans_b, alpha, beta);
    const auto scale = ref::gemm_error_scale(a, b, c, m, n, k, trans_a, trans_b, alpha, beta);
    return test::AllClose(actual, expected, scale, gemm_tolerance(k));
}

// (variant, shape, (trans_a, trans_b), (alpha, beta), offset)
using GemmParam = std::tuple<std::string, Shape, std::pair<bool, bool>, Scalars, size_t>;

// gtest names allow only [A-Za-z0-9_]: 0.5 -> "0p5", -2 -> "m2".
std::string ScalarName(float x) {
    std::string s = std::to_string(x);
    s.erase(s.find_last_not_of('0') + 1);
    if (s.back() == '.') {
        s.pop_back();
    }
    for (char& ch : s) {
        if (ch == '.') {
            ch = 'p';
        }
        if (ch == '-') {
            ch = 'm';
        }
    }
    return s;
}

std::string GemmName(const ::testing::TestParamInfo<GemmParam>& info) {
    const auto& [variant, s, trans, ab, offset] = info.param;
    const std::string ops = std::string(trans.first ? "T" : "N") + (trans.second ? "T" : "N");
    return variant + "_" + std::to_string(s.m) + "x" + std::to_string(s.n) + "x" + std::to_string(s.k) + "_" + ops +
           "_a" + ScalarName(ab.alpha) + "_b" + ScalarName(ab.beta) + "_off" + std::to_string(offset);
}

class GemmKernelTest : public ::testing::TestWithParam<GemmParam> {};

TEST_P(GemmKernelTest, MatchesReferenceWithoutOverrun) {
    const auto& [variant, s, trans, ab, offset] = GetParam();
    const auto [trans_a, trans_b] = trans;
    const GemmFn kernel = kernel_fn(variant);
    const auto host_a = test::random_uniform(s.m * s.k, /*seed=*/s.m * 131 + s.k + offset);
    const auto host_b = test::random_uniform(s.k * s.n, /*seed=*/s.n * 17 + s.k + offset + 1);
    const auto host_c = test::random_uniform(s.m * s.n, /*seed=*/s.m * 7 + s.n + offset + 2);

    test::Guarded a(host_a, offset), b(host_b, offset), c(host_c, offset);
    kernel(a.data(), b.data(), c.data(), s.m, s.n, s.k, trans_a, trans_b, ab.alpha, ab.beta, Context::get().stream());
    Context::get().synchronize();

    EXPECT_TRUE(GemmClose(c.body(), host_a, host_b, host_c, s.m, s.n, s.k, trans_a, trans_b, ab.alpha, ab.beta));
    EXPECT_TRUE(c.guards_intact());
}

INSTANTIATE_TEST_SUITE_P(AllVariants, GemmKernelTest,
                         ::testing::Combine(::testing::ValuesIn(kernels::registry<GemmFn>().variants("gemm")),
                                            ::testing::ValuesIn(kShapes), ::testing::ValuesIn(kTrans),
                                            ::testing::ValuesIn(kScalars), ::testing::ValuesIn(kOffsets)),
                         GemmName);

// Extra tile configs: same shapes / transposes / offsets, one (alpha, beta) since scaling doesn't depend on the tile
// (ctest runs every case in its own process, so the full matrix x every config would blow the job time limit)
INSTANTIATE_TEST_SUITE_P(TileConfigs, GemmKernelTest,
                         ::testing::Combine(::testing::ValuesIn(config_names()), ::testing::ValuesIn(kShapes),
                                            ::testing::ValuesIn(kTrans), ::testing::Values(Scalars{0.5f, -2.0f}),
                                            ::testing::ValuesIn(kOffsets)),
                         GemmName);

// ---------------------------------------------------------------------------- beta == 0

class GemmBetaZeroTest : public ::testing::TestWithParam<std::string> {};

// BLAS semantics: with beta == 0, C is write-only, so NaN already in C must not leak into the result.
TEST_P(GemmBetaZeroTest, IgnoresNaNInC) {
    const std::string& variant = GetParam();
    const GemmFn kernel = kernel_fn(variant);
    const Shape s{37, 29, 19};
    const auto host_a = test::random_uniform(s.m * s.k, /*seed=*/1);
    const auto host_b = test::random_uniform(s.k * s.n, /*seed=*/2);
    const std::vector<float> host_c(s.m * s.n, std::numeric_limits<float>::quiet_NaN());

    test::Guarded a(host_a, 0), b(host_b, 0), c(host_c, 0);
    kernel(a.data(), b.data(), c.data(), s.m, s.n, s.k, false, false, 1.5f, 0.0f, Context::get().stream());
    Context::get().synchronize();

    EXPECT_TRUE(GemmClose(c.body(), host_a, host_b, host_c, s.m, s.n, s.k, false, false, 1.5f, 0.0f));
    EXPECT_TRUE(c.guards_intact());
}

INSTANTIATE_TEST_SUITE_P(AllVariants, GemmBetaZeroTest, ::testing::ValuesIn(kernel_names()));

// ---------------------------------------------------------------------------- mixed alignment

struct Offsets {
    size_t a, b, c;
};

// One operand off 16-byte alignment, then all three off by different amounts
const std::vector<Offsets> kMixedOffsets{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 2, 3}};
// NN with k and n multiples of 4, so vec4 kernels would take the float4 path if every pointer were aligned
const std::vector<Shape> kVec4Shapes{{16, 16, 16}, {17, 36, 20}, {64, 256, 128}};

using MixedParam = std::tuple<std::string, Shape, Offsets>;

std::string MixedName(const ::testing::TestParamInfo<MixedParam>& info) {
    const auto& [variant, s, off] = info.param;
    return variant + "_" + std::to_string(s.m) + "x" + std::to_string(s.n) + "x" + std::to_string(s.k) + "_off" +
           std::to_string(off.a) + std::to_string(off.b) + std::to_string(off.c);
}

class GemmMixedAlignmentTest : public ::testing::TestWithParam<MixedParam> {};

// The launcher must fall back to scalar loads instead of reading through a misaligned float4
TEST_P(GemmMixedAlignmentTest, MatchesReferenceWithoutOverrun) {
    const auto& [variant, s, off] = GetParam();
    const GemmFn kernel = kernel_fn(variant);
    const auto host_a = test::random_uniform(s.m * s.k, /*seed=*/s.m * 131 + s.k);
    const auto host_b = test::random_uniform(s.k * s.n, /*seed=*/s.n * 17 + s.k + 1);
    const auto host_c = test::random_uniform(s.m * s.n, /*seed=*/s.m * 7 + s.n + 2);
    const float alpha = 0.5f, beta = -2.0f;

    test::Guarded a(host_a, off.a), b(host_b, off.b), c(host_c, off.c);
    kernel(a.data(), b.data(), c.data(), s.m, s.n, s.k, false, false, alpha, beta, Context::get().stream());
    Context::get().synchronize();

    EXPECT_TRUE(GemmClose(c.body(), host_a, host_b, host_c, s.m, s.n, s.k, false, false, alpha, beta));
    EXPECT_TRUE(c.guards_intact());
}

INSTANTIATE_TEST_SUITE_P(AllVariants, GemmMixedAlignmentTest,
                         ::testing::Combine(::testing::ValuesIn(kernel_names()), ::testing::ValuesIn(kVec4Shapes),
                                            ::testing::ValuesIn(kMixedOffsets)),
                         MixedName);

// ---------------------------------------------------------------------------- accumulation stress

// Long K piles up the most fp32 rounding, so this is where the gamma_{k+2} * sum |a_i b_i| bound is tightest.
// Positive inputs don't cancel, so the result and its error grow with k.
struct StressCase {
    Shape shape;
    bool positive;  // inputs in [0, 1] instead of [-1, 1]
};

const std::vector<StressCase> kStress{{{32, 32, 1 << 17}, false},
                                      {{32, 32, 1 << 17}, true},
                                      {{256, 384, 8192}, false},
                                      {{256, 384, 8192}, true}};

struct StressData {
    std::vector<float> a, b;
    std::vector<double> expected, scale;
};

// The CPU reference is O(mnk), so each case is built once and shared by every kernel in one process
// (ctest still rebuilds it per test, ~1 s each)
const StressData& stress_data(size_t idx) {
    static std::map<size_t, StressData> cache;
    const auto it = cache.find(idx);
    if (it != cache.end()) return it->second;

    const auto& [s, positive] = kStress[idx];
    const float lo = positive ? 0.0f : -1.0f;
    StressData d;
    d.a = test::random_uniform(s.m * s.k, /*seed=*/idx * 2 + 1, lo, 1.0f);
    d.b = test::random_uniform(s.k * s.n, /*seed=*/idx * 2 + 2, lo, 1.0f);
    const std::vector<float> zero_c(s.m * s.n, 0.0f);
    d.expected = ref::gemm(d.a, d.b, zero_c, s.m, s.n, s.k, false, false, 1.0f, 0.0f);

    // sum |a * b| is the result itself when nothing is negative
    d.scale = positive ? d.expected
                       : ref::gemm_error_scale(d.a, d.b, zero_c, s.m, s.n, s.k, false, false, 1.0f, 0.0f);
    return cache.emplace(idx, std::move(d)).first->second;
}

using StressParam = std::tuple<std::string, size_t>;  // (variant, index into kStress)

std::string StressName(const ::testing::TestParamInfo<StressParam>& info) {
    const auto& [variant, idx] = info.param;
    const auto& [s, positive] = kStress[idx];
    return variant + "_" + std::to_string(s.m) + "x" + std::to_string(s.n) + "x" + std::to_string(s.k) +
           (positive ? "_positive" : "_signed");
}

class GemmAccumStressTest : public ::testing::TestWithParam<StressParam> {};

TEST_P(GemmAccumStressTest, StaysUnderDotProductBound) {
    const auto& [variant, idx] = GetParam();
    const Shape s = kStress[idx].shape;
    const StressData& d = stress_data(idx);

    test::Guarded a(d.a, 0), b(d.b, 0), c(s.m * s.n, 0);
    kernel_fn(variant)(a.data(), b.data(), c.data(), s.m, s.n, s.k, false, false, 1.0f, 0.0f,
                       Context::get().stream());
    Context::get().synchronize();

    // max_rel is max err / sum|a b|; dividing by gamma gives how much of the worst-case bound is used
    const auto out = c.body();
    const test::Tolerance tol = gemm_tolerance(s.k);
    const test::CompareReport r = test::compare(out.data(), d.expected.data(), out.size(), tol, d.scale.data());
    std::printf("[ info ] max err/bound %.3e  %s\n", r.max_rel / tol.rtol, r.summary().c_str());
    EXPECT_TRUE(r.ok()) << r.summary() << " exceeds the gamma_{k+2} sum|a b| bound";
    EXPECT_TRUE(c.guards_intact());
}

INSTANTIATE_TEST_SUITE_P(AllVariants, GemmAccumStressTest,
                         ::testing::Combine(::testing::ValuesIn(kernel_names()),
                                            ::testing::Range(size_t{0}, kStress.size())),
                         StressName);

}  // namespace
}  // namespace cudl
