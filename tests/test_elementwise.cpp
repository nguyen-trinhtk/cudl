// Elementwise kernels: every registered op x variant x size x alignment offset, checked against
// the CPU reference, plus guard checks and IEEE edge cases (NaN, inf, signed zero).

#include <gtest/gtest.h>

#include <cmath>
#include <iterator>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include "common/compare.h"
#include "common/guarded.h"
#include "common/random.h"
#include "cudl/core/context.h"
#include "cudl/kernels/elementwise.h"
#include "cudl/kernels/registry.h"
#include "ref/elementwise_ref.h"

namespace cudl {
namespace {

// Op name lists as std::string, so they combine with the other string parameters.
template <size_t N>
std::vector<std::string> op_names(const char* const (&ops)[N]) {
    return {std::begin(ops), std::end(ops)};
}

using kernels::BiasFn;
using kernels::BinaryFn;
using kernels::ScaleFn;
using kernels::UnaryFn;

constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

const std::vector<size_t> kSizes{0, 1, 3, 4, 5, 255, 257, 4099, (1 << 20) + 3};
const std::vector<size_t> kOffsets{0, 1, 3};  // 1 and 3 break 16-byte alignment

// neg/relu and correctly rounded arithmetic are exact; transcendentals are within a few ULP.
test::Tolerance unary_tolerance(const std::string& op) {
    if (op == "neg" || op == "relu") return test::kExact;
    return {1e-6, 1e-5, 4};
}

test::Tolerance binary_tolerance(const std::string& op) {
    // Rounding the double quotient to float can differ from fp32 division by one ULP.
    if (op == "div") return {0.0, 0.0, 1};
    return test::kExact;
}

// ---------------------------------------------------------------------------- unary

using EwParam = std::tuple<std::string, std::string, size_t, size_t>;  // (op, variant, n, offset)

std::string EwName(const ::testing::TestParamInfo<EwParam>& info) {
    const auto& [op, variant, n, offset] = info.param;
    return op + "_" + variant + "_n" + std::to_string(n) + "_off" + std::to_string(offset);
}

class UnaryKernelTest : public ::testing::TestWithParam<EwParam> {};

TEST_P(UnaryKernelTest, MatchesReferenceWithoutOverrun) {
    const auto& [op, variant, n, offset] = GetParam();
    const auto& kernel = kernels::registry<UnaryFn>().get(op, variant);
    const float range = op == "exp" ? 10.0f : 5.0f;
    const auto host_in = test::random_uniform(n, /*seed=*/n * 31 + offset, -range, range);

    test::Guarded in(host_in, offset), out(n, offset);
    kernel.fn(in.data(), out.data(), n, Context::get().stream());
    Context::get().synchronize();

    EXPECT_TRUE(test::AllClose(out.body(), ref::unary(op, host_in), unary_tolerance(op)));
    EXPECT_TRUE(out.guards_intact());
}

INSTANTIATE_TEST_SUITE_P(AllVariants, UnaryKernelTest,
                         ::testing::Combine(::testing::ValuesIn(op_names(kernels::kUnaryOps)),
                                            ::testing::ValuesIn(kernels::registry<UnaryFn>().variants("relu")),
                                            ::testing::ValuesIn(kSizes), ::testing::ValuesIn(kOffsets)),
                         EwName);

// ---------------------------------------------------------------------------- binary

class BinaryKernelTest : public ::testing::TestWithParam<EwParam> {};

TEST_P(BinaryKernelTest, MatchesReferenceWithoutOverrun) {
    const auto& [op, variant, n, offset] = GetParam();
    const auto& kernel = kernels::registry<BinaryFn>().get(op, variant);
    const auto host_a = test::random_uniform(n, /*seed=*/n * 31 + offset);
    // Keep divisors away from zero.
    const auto host_b = op == "div" ? test::random_uniform(n, n * 17 + offset + 1, 0.5f, 2.0f)
                                    : test::random_uniform(n, n * 17 + offset + 1);

    test::Guarded a(host_a, offset), b(host_b, offset), out(n, offset);
    kernel.fn(a.data(), b.data(), out.data(), n, Context::get().stream());
    Context::get().synchronize();

    EXPECT_TRUE(test::AllClose(out.body(), ref::binary(op, host_a, host_b), binary_tolerance(op)));
    EXPECT_TRUE(out.guards_intact());
}

INSTANTIATE_TEST_SUITE_P(AllVariants, BinaryKernelTest,
                         ::testing::Combine(::testing::ValuesIn(op_names(kernels::kBinaryOps)),
                                            ::testing::ValuesIn(kernels::registry<BinaryFn>().variants("add")),
                                            ::testing::ValuesIn(kSizes), ::testing::ValuesIn(kOffsets)),
                         EwName);

// ---------------------------------------------------------------------------- scale

using ScaleParam = std::tuple<std::string, size_t, size_t>;  // (variant, n, offset)

class ScaleKernelTest : public ::testing::TestWithParam<ScaleParam> {};

std::string ScaleName(const ::testing::TestParamInfo<ScaleParam>& info) {
    const auto& [variant, n, offset] = info.param;
    return variant + "_n" + std::to_string(n) + "_off" + std::to_string(offset);
}

TEST_P(ScaleKernelTest, MatchesReferenceWithoutOverrun) {
    const auto& [variant, n, offset] = GetParam();
    const auto& kernel = kernels::registry<ScaleFn>().get("scale", variant);
    const float alpha = 0.37f;
    const auto host_in = test::random_uniform(n, /*seed=*/n * 31 + offset);

    test::Guarded in(host_in, offset), out(n, offset);
    kernel.fn(in.data(), alpha, out.data(), n, Context::get().stream());
    Context::get().synchronize();

    EXPECT_TRUE(test::AllClose(out.body(), ref::scale(host_in, alpha), test::kExact));
    EXPECT_TRUE(out.guards_intact());
}

INSTANTIATE_TEST_SUITE_P(AllVariants, ScaleKernelTest,
                         ::testing::Combine(::testing::ValuesIn(kernels::registry<ScaleFn>().variants("scale")),
                                            ::testing::ValuesIn(kSizes), ::testing::ValuesIn(kOffsets)),
                         ScaleName);

// ---------------------------------------------------------------------------- bias broadcast

using BiasParam = std::tuple<std::string, std::pair<size_t, size_t>, size_t>;  // (variant, (rows, cols), offset)

class BiasKernelTest : public ::testing::TestWithParam<BiasParam> {};

std::string BiasName(const ::testing::TestParamInfo<BiasParam>& info) {
    const auto& [variant, shape, offset] = info.param;
    return variant + "_" + std::to_string(shape.first) + "x" + std::to_string(shape.second) + "_off" +
           std::to_string(offset);
}

TEST_P(BiasKernelTest, MatchesReferenceWithoutOverrun) {
    const auto& [variant, shape, offset] = GetParam();
    const auto [rows, cols] = shape;
    const auto& kernel = kernels::registry<BiasFn>().get("add_bias", variant);
    const auto host_x = test::random_uniform(rows * cols, /*seed=*/rows * 131 + cols + offset);
    const auto host_bias = test::random_uniform(cols, /*seed=*/cols * 7 + offset + 1);

    test::Guarded x(host_x, offset), bias(host_bias, offset), out(rows * cols, offset);
    kernel.fn(x.data(), bias.data(), out.data(), rows, cols, Context::get().stream());
    Context::get().synchronize();

    EXPECT_TRUE(test::AllClose(out.body(), ref::add_bias(host_x, host_bias, rows, cols), test::kExact));
    EXPECT_TRUE(out.guards_intact());
}

// cols % 4 != 0 exercises the vec4 fallback even when aligned.
INSTANTIATE_TEST_SUITE_P(
    AllVariants, BiasKernelTest,
    ::testing::Combine(::testing::ValuesIn(kernels::registry<BiasFn>().variants("add_bias")),
                       ::testing::ValuesIn(std::vector<std::pair<size_t, size_t>>{
                           {0, 4}, {4, 0}, {1, 1}, {3, 5}, {7, 4}, {64, 256}, {129, 1000}, {1000, 1024}}),
                       ::testing::ValuesIn(kOffsets)),
    BiasName);

// ---------------------------------------------------------------------------- edge cases

class UnaryEdgeTest : public ::testing::TestWithParam<std::string> {};

std::vector<float> run_unary(const std::string& op, const std::string& variant, const std::vector<float>& in) {
    test::Guarded d_in(in, 0), d_out(in.size(), 0);
    kernels::registry<UnaryFn>().get(op, variant).fn(d_in.data(), d_out.data(), in.size(), Context::get().stream());
    Context::get().synchronize();
    return d_out.body();
}

TEST_P(UnaryEdgeTest, NonFiniteAndSignedZero) {
    const std::string& variant = GetParam();
    const std::vector<float> in{kNaN, kInf, -kInf, -0.0f, 0.0f, 100.0f, -100.0f, 1.0f};

    const auto relu = run_unary("relu", variant, in);
    EXPECT_TRUE(std::isnan(relu[0])) << "relu should propagate NaN";
    EXPECT_TRUE(test::AllClose({relu.begin() + 1, relu.end()}, {kInf, 0, -0.0, 0, 100, 0, 1}, test::kExact));

    const auto e = run_unary("exp", variant, in);
    EXPECT_TRUE(std::isnan(e[0]));
    EXPECT_EQ(e[1], kInf);
    EXPECT_EQ(e[2], 0.0f);
    EXPECT_EQ(e[3], 1.0f);
    EXPECT_EQ(e[5], kInf) << "exp(100) overflows fp32";

    // sigmoid and tanh saturate instead of producing NaN.
    const auto s = run_unary("sigmoid", variant, in);
    EXPECT_EQ(s[1], 1.0f);
    EXPECT_EQ(s[2], 0.0f);
    EXPECT_EQ(s[5], 1.0f);
    EXPECT_EQ(s[6], 0.0f);
    const auto t = run_unary("tanh", variant, in);
    EXPECT_EQ(t[1], 1.0f);
    EXPECT_EQ(t[2], -1.0f);
    EXPECT_EQ(t[5], 1.0f);
    EXPECT_EQ(t[6], -1.0f);
}

INSTANTIATE_TEST_SUITE_P(AllVariants, UnaryEdgeTest,
                         ::testing::ValuesIn(kernels::registry<UnaryFn>().variants("relu")));

}  // namespace
}  // namespace cudl
