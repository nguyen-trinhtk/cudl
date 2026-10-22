// GEMM kernels: every registered variant x shape x transposes x (alpha, beta) x alignment offset, checked against the
// CPU reference plus guard checks, and beta == 0 must not read C.

#include <gtest/gtest.h>

#include <limits>
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

// Includes empty dims, K == 0, and shapes that don't divide the tile
const std::vector<Shape> kShapes{{0, 4, 4},   {4, 0, 4},     {3, 5, 0},      {1, 1, 1},       {1, 7, 13},
                                 {16, 16, 16}, {17, 33, 9},  {127, 65, 33},  {64, 256, 128},  {300, 200, 513}};
const std::vector<Scalars> kScalars{{1.0f, 0.0f}, {1.0f, 1.0f}, {0.5f, -2.0f}, {0.0f, 0.75f}};
const std::vector<size_t> kOffsets{0, 1};
const std::vector<std::pair<bool, bool>> kTrans{{false, false}, {true, false}, {false, true}, {true, true}};

// fp32 dot-product error grows with K (inputs are in [-1, 1])
test::Tolerance gemm_tolerance(size_t k) {
    return {1e-6 * static_cast<double>(k + 1), 1e-5, -1};
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
    const auto& kernel = kernels::registry<GemmFn>().get("gemm", variant);
    const auto host_a = test::random_uniform(s.m * s.k, /*seed=*/s.m * 131 + s.k + offset);
    const auto host_b = test::random_uniform(s.k * s.n, /*seed=*/s.n * 17 + s.k + offset + 1);
    const auto host_c = test::random_uniform(s.m * s.n, /*seed=*/s.m * 7 + s.n + offset + 2);

    test::Guarded a(host_a, offset), b(host_b, offset), c(host_c, offset);
    kernel.fn(a.data(), b.data(), c.data(), s.m, s.n, s.k, trans_a, trans_b, ab.alpha, ab.beta,
              Context::get().stream());
    Context::get().synchronize();

    const auto expected = ref::gemm(host_a, host_b, host_c, s.m, s.n, s.k, trans_a, trans_b, ab.alpha, ab.beta);
    EXPECT_TRUE(test::AllClose(c.body(), expected, gemm_tolerance(s.k)));
    EXPECT_TRUE(c.guards_intact());
}

INSTANTIATE_TEST_SUITE_P(AllVariants, GemmKernelTest,
                         ::testing::Combine(::testing::ValuesIn(kernels::registry<GemmFn>().variants("gemm")),
                                            ::testing::ValuesIn(kShapes), ::testing::ValuesIn(kTrans),
                                            ::testing::ValuesIn(kScalars), ::testing::ValuesIn(kOffsets)),
                         GemmName);

// ---------------------------------------------------------------------------- beta == 0

class GemmBetaZeroTest : public ::testing::TestWithParam<std::string> {};

// BLAS semantics: with beta == 0, C is write-only, so NaN already in C must not leak into the result.
TEST_P(GemmBetaZeroTest, IgnoresNaNInC) {
    const std::string& variant = GetParam();
    const auto& kernel = kernels::registry<GemmFn>().get("gemm", variant);
    const Shape s{37, 29, 19};
    const auto host_a = test::random_uniform(s.m * s.k, /*seed=*/1);
    const auto host_b = test::random_uniform(s.k * s.n, /*seed=*/2);
    const std::vector<float> host_c(s.m * s.n, std::numeric_limits<float>::quiet_NaN());

    test::Guarded a(host_a, 0), b(host_b, 0), c(host_c, 0);
    kernel.fn(a.data(), b.data(), c.data(), s.m, s.n, s.k, false, false, 1.5f, 0.0f, Context::get().stream());
    Context::get().synchronize();

    const auto expected = ref::gemm(host_a, host_b, host_c, s.m, s.n, s.k, false, false, 1.5f, 0.0f);
    EXPECT_TRUE(test::AllClose(c.body(), expected, gemm_tolerance(s.k)));
    EXPECT_TRUE(c.guards_intact());
}

INSTANTIATE_TEST_SUITE_P(AllVariants, GemmBetaZeroTest,
                         ::testing::ValuesIn(kernels::registry<GemmFn>().variants("gemm")));

}  // namespace
}  // namespace cudl
