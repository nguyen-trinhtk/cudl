// Fused softmax cross-entropy kernels: every registered variant x shape, checked against the double CPU
// reference, plus overflow edge cases (huge logits) and out-of-range labels.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "common/compare.h"
#include "common/guarded.h"
#include "common/random.h"
#include "cudl/core/context.h"
#include "cudl/core/tensor.h"
#include "cudl/kernels/loss.h"
#include "cudl/kernels/registry.h"
#include "ref/loss_ref.h"

namespace cudl {
namespace {

using kernels::CrossEntropyLossFn;

// exp/log are within a few ULP; the loss is a difference of O(log cols) terms.
constexpr test::Tolerance kCrossEntropyTol{1e-5, 1e-5, -1};
// With logits near 1000 one float ULP is 6e-5, and lse - x loses that much absolute precision.
constexpr test::Tolerance kLargeLogitTol{2e-4, 1e-5, -1};

std::vector<int32_t> random_labels(size_t rows, size_t cols, uint64_t seed) {
    const auto u = test::random_uniform(rows, seed, 0.0f, static_cast<float>(cols));
    std::vector<int32_t> out(rows);
    for (size_t i = 0; i < rows; ++i) {
        const int32_t label = static_cast<int32_t>(u[i]);
        out[i] = label < static_cast<int32_t>(cols) ? label : static_cast<int32_t>(cols) - 1;
    }
    return out;
}

struct CrossEntropyRun {
    std::vector<float> loss, dlogits;
};

CrossEntropyRun run_cross_entropy(const std::string& variant, const std::vector<float>& logits,
                                  const std::vector<int32_t>& labels, size_t rows, size_t cols, float grad_scale) {
    const auto& k = kernels::registry<CrossEntropyLossFn>().get("cross_entropy_loss", variant);
    const Tensor d_labels = Tensor::from_vector(labels, {static_cast<int64_t>(rows)});
    test::Guarded d_logits(logits, 1), loss(rows, 1), dlogits(rows * cols, 1);
    k.fn(d_logits.data(), d_labels.data<int32_t>(), loss.data(), dlogits.data(), rows, cols, grad_scale,
         Context::get().stream());
    Context::get().synchronize();
    EXPECT_TRUE(loss.guards_intact());
    EXPECT_TRUE(dlogits.guards_intact());
    return {loss.body(), dlogits.body()};
}

// ---------------------------------------------------------------------------- random logits

using CrossEntropyParam = std::tuple<std::string, std::pair<size_t, size_t>>;  // (variant, (rows, cols))

class CrossEntropyLossTest : public ::testing::TestWithParam<CrossEntropyParam> {};

std::string CrossEntropyName(const ::testing::TestParamInfo<CrossEntropyParam>& info) {
    const auto& [variant, shape] = info.param;
    return variant + "_" + std::to_string(shape.first) + "x" + std::to_string(shape.second);
}

TEST_P(CrossEntropyLossTest, MatchesReferenceWithoutOverrun) {
    const auto& [variant, shape] = GetParam();
    const auto [rows, cols] = shape;
    const auto logits = test::random_uniform(rows * cols, /*seed=*/rows * 131 + cols, -8.0f, 8.0f);
    const auto labels = random_labels(rows, cols, /*seed=*/rows + cols);
    const float scale = rows ? 1.0f / static_cast<float>(rows) : 1.0f;

    const auto got = run_cross_entropy(variant, logits, labels, rows, cols, scale);
    const auto expected = ref::cross_entropy_loss(logits, labels, rows, cols, scale);
    EXPECT_TRUE(test::AllClose(got.loss, expected.loss, kCrossEntropyTol));
    EXPECT_TRUE(test::AllClose(got.dlogits, expected.dlogits, kCrossEntropyTol));
}

INSTANTIATE_TEST_SUITE_P(
    AllVariants, CrossEntropyLossTest,
    ::testing::Combine(::testing::ValuesIn(kernels::registry<CrossEntropyLossFn>().variants("cross_entropy_loss")),
                       ::testing::ValuesIn(std::vector<std::pair<size_t, size_t>>{
                           {0, 10}, {1, 1}, {1, 10}, {7, 31}, {33, 32}, {128, 10}, {100, 257}, {3, 5000}})),
    CrossEntropyName);

// ---------------------------------------------------------------------------- edge cases

class CrossEntropyLossEdgeTest : public ::testing::TestWithParam<std::string> {};

// Without max-subtraction exp(1000) overflows to inf and the loss becomes NaN.
TEST_P(CrossEntropyLossEdgeTest, LargeLogitsDoNotOverflow) {
    const size_t rows = 3, cols = 4;
    const std::vector<float> logits{1000.0f, 0.0f,    -1000.0f, 999.0f,   // confident, label is the max
                                    -1000.0f, -999.0f, -1001.0f, -1000.0f,  // all very negative
                                    88.0f,    89.0f,   90.0f,    -88.0f};  // near expf's overflow point
    const std::vector<int32_t> labels{0, 3, 3};

    const auto got = run_cross_entropy(GetParam(), logits, labels, rows, cols, 1.0f);
    const auto expected = ref::cross_entropy_loss(logits, labels, rows, cols, 1.0f);
    for (float v : got.loss) {
        EXPECT_TRUE(std::isfinite(v));
    }
    EXPECT_TRUE(test::AllClose(got.loss, expected.loss, kLargeLogitTol));
    EXPECT_TRUE(test::AllClose(got.dlogits, expected.dlogits, kLargeLogitTol));
}

TEST_P(CrossEntropyLossEdgeTest, OutOfRangeLabelGivesNaN) {
    const size_t rows = 3, cols = 4;
    const auto logits = test::random_uniform(rows * cols, /*seed=*/3);
    const std::vector<int32_t> labels{1, -1, 4};

    const auto got = run_cross_entropy(GetParam(), logits, labels, rows, cols, 1.0f);
    EXPECT_TRUE(std::isfinite(got.loss[0]));
    EXPECT_TRUE(std::isnan(got.loss[1]));
    EXPECT_TRUE(std::isnan(got.loss[2]));
    const auto expected = ref::cross_entropy_loss(logits, labels, rows, cols, 1.0f);
    EXPECT_TRUE(test::AllClose(got.dlogits, expected.dlogits, kCrossEntropyTol));
}

INSTANTIATE_TEST_SUITE_P(AllVariants, CrossEntropyLossEdgeTest,
                         ::testing::ValuesIn(kernels::registry<CrossEntropyLossFn>().variants("cross_entropy_loss")));

}  // namespace
}  // namespace cudl
