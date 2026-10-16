// Copy kernels: every registered variant x size x alignment offset, checked against
// the CPU reference, plus guard checks that kernels don't write outside their range.

#include <gtest/gtest.h>

#include <string>
#include <tuple>
#include <vector>

#include "common/compare.h"
#include "common/random.h"
#include "cudl/core/context.h"
#include "cudl/core/tensor.h"
#include "cudl/kernels/copy.h"
#include "cudl/kernels/registry.h"
#include "ref/copy_ref.h"

namespace cudl {
namespace {

using kernels::CopyFn;

constexpr float kSentinel = -12345.0f;
constexpr size_t kGuard = 7;

// (variant, n, element offset). Offsets 1 and 3 break 16-byte alignment.
using CopyParam = std::tuple<std::string, size_t, size_t>;

class CopyKernelTest : public ::testing::TestWithParam<CopyParam> {};

TEST_P(CopyKernelTest, MatchesReferenceWithoutOverrun) {
    const auto& [variant, n, offset] = GetParam();
    const auto& kernel = kernels::registry<CopyFn>().get("copy", variant);
    const size_t total = offset + n + kGuard;

    const auto host_src = test::random_uniform(total, /*seed=*/n * 31 + offset);
    Tensor src = Tensor::from_vector(host_src, {static_cast<int64_t>(total)});
    Tensor dst = Tensor::from_vector(std::vector<float>(total, kSentinel), {static_cast<int64_t>(total)});

    kernel.fn(src.data<float>() + offset, dst.data<float>() + offset, n, Context::get().stream());
    Context::get().synchronize();

    const auto out = dst.to_vector();
    const std::vector<float> body(out.begin() + offset, out.begin() + offset + n);
    const std::vector<float> src_body(host_src.begin() + offset, host_src.begin() + offset + n);
    EXPECT_TRUE(test::AllClose(body, ref::copy(src_body), test::kExact));

    for (size_t i = 0; i < offset; ++i) EXPECT_EQ(out[i], kSentinel) << "wrote before start at " << i;
    for (size_t i = offset + n; i < total; ++i) EXPECT_EQ(out[i], kSentinel) << "wrote past end at " << i;
}

std::string ParamName(const ::testing::TestParamInfo<CopyParam>& info) {
    const auto& [variant, n, offset] = info.param;
    return variant + "_n" + std::to_string(n) + "_off" + std::to_string(offset);
}

// Every registered variant is tested automatically.
INSTANTIATE_TEST_SUITE_P(AllVariants, CopyKernelTest,
                         ::testing::Combine(::testing::ValuesIn(kernels::registry<CopyFn>().variants("copy")),
                                            ::testing::ValuesIn(std::vector<size_t>{0, 1, 3, 4, 5, 255, 256, 257, 4099,
                                                                                    1 << 20, (1 << 20) + 3}),
                                            ::testing::ValuesIn(std::vector<size_t>{0, 1, 3})),
                         ParamName);

TEST(KernelRegistry, UnknownVariantThrows) {
    EXPECT_THROW(kernels::registry<CopyFn>().get("copy", "nope"), std::runtime_error);
}

TEST(KernelRegistry, UnknownOpThrows) { EXPECT_THROW(kernels::registry<CopyFn>().entries("nope"), std::runtime_error); }

}  // namespace
}  // namespace cudl
