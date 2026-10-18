// Reduction kernels: every registered op x variant x size, checked against the double CPU
// reference, plus NaN handling, a determinism check and the summation drift sweep.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include "common/compare.h"
#include "common/guarded.h"
#include "common/random.h"
#include "cudl/core/context.h"
#include "cudl/kernels/reduce.h"
#include "cudl/kernels/registry.h"
#include "ref/reduce_ref.h"

namespace cudl {
namespace {

// Op name lists as std::string, so they combine with the other string parameters.
template <size_t N>
std::vector<std::string> op_names(const char* const (&ops)[N]) {
    return {std::begin(ops), std::end(ops)};
}

using kernels::kReduceWorkspaceFloats;
using kernels::ReduceAllFn;
using kernels::ReduceRowsFn;

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

// Random integers in [-8, 8]: every partial sum is exactly representable, so any summation
// order gives the exact answer and a dropped or double-counted element always shows up.
std::vector<float> random_integers(size_t n, uint64_t seed) {
    auto v = test::random_uniform(n, seed, -8.5f, 8.5f);
    for (float& x : v) x = std::round(x);
    return v;
}

// Rounding error of a float sum is bounded relative to sum(|x|), not to the (possibly tiny) sum.
test::Tolerance sum_tolerance(const std::vector<float>& in) {
    double sum_abs = 0.0;
    for (float x : in) sum_abs += std::fabs(x);
    return {1e-6 * sum_abs + 1e-6, 0.0, -1};
}

float run_reduce_all(const kernels::KernelEntry<ReduceAllFn>& k, const std::vector<float>& in, size_t offset = 0) {
    test::Guarded d_in(in, offset), out(1, offset), workspace(kReduceWorkspaceFloats, offset);
    k.fn(d_in.data(), out.data(), in.size(), workspace.data(), Context::get().stream());
    Context::get().synchronize();
    EXPECT_TRUE(out.guards_intact());
    EXPECT_TRUE(workspace.guards_intact()) << "wrote past the workspace";
    return out.body()[0];
}

// ---------------------------------------------------------------------------- reduce all

using AllParam = std::tuple<std::string, std::string, size_t, size_t>;  // (op, variant, n, offset)

class ReduceAllTest : public ::testing::TestWithParam<AllParam> {};

std::string AllName(const ::testing::TestParamInfo<AllParam>& info) {
    const auto& [op, variant, n, offset] = info.param;
    return op + "_" + variant + "_n" + std::to_string(n) + "_off" + std::to_string(offset);
}

TEST_P(ReduceAllTest, ExactOnIntegers) {
    const auto& [op, variant, n, offset] = GetParam();
    const auto in = random_integers(n, /*seed=*/n * 31 + offset);
    const float got = run_reduce_all(kernels::registry<ReduceAllFn>().get(op, variant), in, offset);
    EXPECT_TRUE(test::AllClose({got}, ref::reduce_all(op, in), test::kExact));
}

TEST_P(ReduceAllTest, MatchesReferenceOnFloats) {
    const auto& [op, variant, n, offset] = GetParam();
    const auto in = test::random_uniform(n, /*seed=*/n * 17 + offset + 1);
    const float got = run_reduce_all(kernels::registry<ReduceAllFn>().get(op, variant), in, offset);
    EXPECT_TRUE(test::AllClose({got}, ref::reduce_all(op, in), op == "max" ? test::kExact : sum_tolerance(in)));
}

INSTANTIATE_TEST_SUITE_P(AllVariants, ReduceAllTest,
                         ::testing::Combine(::testing::ValuesIn(op_names(kernels::kReduceOps)),
                                            ::testing::ValuesIn(kernels::registry<ReduceAllFn>().variants("sum")),
                                            ::testing::ValuesIn(std::vector<size_t>{0, 1, 3, 31, 32, 33, 255, 256, 257,
                                                                                    1000, 4099, 1 << 16, (1 << 20) + 3,
                                                                                    (1 << 22) + 1}),
                                            ::testing::ValuesIn(std::vector<size_t>{0, 1})),
                         AllName);

class ReduceAllMaxTest : public ::testing::TestWithParam<std::string> {};

TEST_P(ReduceAllMaxTest, IgnoresNaN) {
    const auto& k = kernels::registry<ReduceAllFn>().get("max", GetParam());
    auto in = test::random_uniform(100003, /*seed=*/5);
    for (size_t i = 0; i < in.size(); i += 97) in[i] = kNaN;
    EXPECT_TRUE(test::AllClose({run_reduce_all(k, in)}, ref::reduce_all("max", in), test::kExact));

    const std::vector<float> all_nan(1000, kNaN);
    EXPECT_EQ(run_reduce_all(k, all_nan), -std::numeric_limits<float>::infinity());
}

INSTANTIATE_TEST_SUITE_P(AllVariants, ReduceAllMaxTest,
                         ::testing::ValuesIn(kernels::registry<ReduceAllFn>().variants("max")));

// ---------------------------------------------------------------------------- reduce rows

using RowsParam = std::tuple<std::string, std::string, std::pair<size_t, size_t>>;  // (op, variant, (rows, cols))

class ReduceRowsTest : public ::testing::TestWithParam<RowsParam> {};

std::string RowsName(const ::testing::TestParamInfo<RowsParam>& info) {
    const auto& [op, variant, shape] = info.param;
    return op + "_" + variant + "_" + std::to_string(shape.first) + "x" + std::to_string(shape.second);
}

std::vector<float> run_reduce_rows(const std::string& op, const std::string& variant, const std::vector<float>& in,
                                   size_t rows, size_t cols) {
    test::Guarded d_in(in, 1), out(rows, 1);
    kernels::registry<ReduceRowsFn>().get(op, variant).fn(d_in.data(), out.data(), rows, cols, Context::get().stream());
    Context::get().synchronize();
    EXPECT_TRUE(out.guards_intact());
    return out.body();
}

TEST_P(ReduceRowsTest, ExactOnIntegers) {
    const auto& [op, variant, shape] = GetParam();
    const auto [rows, cols] = shape;
    const auto in = random_integers(rows * cols, /*seed=*/rows * 131 + cols);
    EXPECT_TRUE(test::AllClose(run_reduce_rows(op, variant, in, rows, cols), ref::reduce_rows(op, in, rows, cols),
                               test::kExact));
}

TEST_P(ReduceRowsTest, MatchesReferenceOnFloats) {
    const auto& [op, variant, shape] = GetParam();
    const auto [rows, cols] = shape;
    const auto in = test::random_uniform(rows * cols, /*seed=*/rows * 17 + cols + 1);
    // Rows are short compared to the reduce-all sizes; a uniform tolerance of cols * eps is ample.
    const test::Tolerance tol = op == "max" ? test::kExact : test::Tolerance{1e-6 + cols * 1e-7, 0.0, -1};
    EXPECT_TRUE(
        test::AllClose(run_reduce_rows(op, variant, in, rows, cols), ref::reduce_rows(op, in, rows, cols), tol));
}

INSTANTIATE_TEST_SUITE_P(
    AllVariants, ReduceRowsTest,
    ::testing::Combine(
        ::testing::ValuesIn(op_names(kernels::kReduceOps)),
        ::testing::ValuesIn(kernels::registry<ReduceRowsFn>().variants("sum")),
        ::testing::ValuesIn(std::vector<std::pair<size_t, size_t>>{
            {0, 5}, {3, 0}, {1, 1}, {1, 1000}, {7, 31}, {33, 32}, {100, 257}, {1000, 1024}, {4, 100000}})),
    RowsName);

// ---------------------------------------------------------------------------- determinism

// Variants registered as deterministic must give bit-identical results on every run. The
// non-deterministic ones (atomic sum) are only reported.
TEST(ReduceDeterminism, RepeatedRunsAreBitIdentical) {
    constexpr int kRuns = 10;
    const auto in = test::random_uniform(size_t{1} << 22, /*seed=*/99);
    for (const char* op : kernels::kReduceOps) {
        for (const auto& k : kernels::registry<ReduceAllFn>().entries(op)) {
            const float first = run_reduce_all(k, in);
            int differing = 0;
            for (int i = 1; i < kRuns; ++i) {
                const float again = run_reduce_all(k, in);
                if (std::memcmp(&first, &again, sizeof first) != 0) ++differing;
            }
            if (k.deterministic) {
                EXPECT_EQ(differing, 0) << op << "/" << k.variant << " is registered as deterministic";
            } else {
                std::printf("[ info ] %s/%s (non-deterministic): %d of %d runs differed from the first\n", op,
                            k.variant.c_str(), differing, kRuns - 1);
            }
        }
    }
}

// ---------------------------------------------------------------------------- drift

// Relative error of fp32 summation against the double reference as N grows, for every sum
// variant plus two CPU fp32 baselines (sequential and pairwise). Inputs are in [0, 1), so the
// sum grows with N and sequential summation visibly loses precision.
// Set CUDL_DRIFT_CSV=path to save the table; scripts/plot_drift.py plots it.
TEST(ReduceDrift, SumErrorVsN) {
    const auto& entries = kernels::registry<ReduceAllFn>().entries("sum");
    std::vector<std::string> names{"cpu_sequential", "cpu_pairwise"};
    for (const auto& k : entries) names.push_back(k.variant);

    std::string csv = "n";
    for (const auto& name : names) csv += "," + name;
    csv += "\n";
    std::printf("%10s", "n");
    for (const auto& name : names) std::printf(" %15s", name.c_str());
    std::printf("\n");

    for (int log_n = 10; log_n <= 24; log_n += 2) {
        const size_t n = size_t{1} << log_n;
        const auto in = test::random_uniform(n, /*seed=*/log_n, 0.0f, 1.0f);
        const double expected = ref::reduce("sum", in.data(), n);
        auto rel = [&](float got) { return std::fabs(static_cast<double>(got) - expected) / expected; };

        std::vector<double> errs{rel(ref::sum_f32_sequential(in)), rel(ref::sum_f32_pairwise(in.data(), n))};
        for (const auto& k : entries) errs.push_back(rel(run_reduce_all(k, in)));

        csv += std::to_string(n);
        std::printf("%10zu", n);
        for (size_t i = 0; i < errs.size(); ++i) {
            char buf[32];
            std::snprintf(buf, sizeof buf, "%.3e", errs[i]);
            csv += std::string(",") + buf;
            std::printf(" %15s", buf);
            // Tree reductions keep error far below sequential summation at every size.
            const bool tree = names[i].rfind("smem", 0) == 0 || names[i].rfind("warp", 0) == 0;
            if (tree) EXPECT_LT(errs[i], 1e-5) << names[i] << " at n=" << n;
        }
        csv += "\n";
        std::printf("\n");
    }

    if (const char* path = std::getenv("CUDL_DRIFT_CSV")) {
        std::ofstream(path) << csv;
        std::printf("wrote %s\n", path);
    }
}

}  // namespace
}  // namespace cudl
