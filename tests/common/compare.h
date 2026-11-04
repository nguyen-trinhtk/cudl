#pragma once

// Compares GPU output against a double-precision reference. Reports max absolute,
// relative and ULP error plus the worst element; this is the FP drift measurement
// every kernel test uses.

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

namespace cudl::test {

// An element passes when |actual - expected| <= atol + rtol * scale, where scale = |expected| by default,
// or (if max_ulp >= 0) when its ULP distance to float(expected) is at most max_ulp.
// NaN matches NaN; infinities must match exactly.
struct Tolerance {
    double atol = 0.0;
    double rtol = 0.0;
    int64_t max_ulp = -1;
};

inline constexpr Tolerance kExact{0.0, 0.0, 0};
inline constexpr Tolerance kF32Default{1e-6, 1e-5, -1};

struct CompareReport {
    size_t n = 0;
    size_t mismatches = 0;
    double max_abs = 0.0;
    double max_rel = 0.0;  // max |actual - expected| / scale
    int64_t max_ulp = 0;
    size_t worst_index = 0;  // index of max_abs
    float worst_actual = 0.0f;
    double worst_expected = 0.0;

    bool ok() const { return mismatches == 0; }
    std::string summary() const;
};

// `expected` is double: CPU references accumulate in double, so the reference
// itself does not add fp32 rounding error.
// `scale` (optional, n entries) replaces |expected| as the magnitude rtol is relative to. Use it when the
// rounding-error bound isn't proportional to the result, e.g. a dot product's error scales with sum |a_i b_i|,
// not |sum a_i b_i|, so under cancellation |expected| is ~0 while the error isn't (see ref::gemm_error_scale).
CompareReport compare(const float* actual, const double* expected, size_t n, const Tolerance& tol,
                      const double* scale = nullptr);

// Use as EXPECT_TRUE(AllClose(...)); the failure message carries the full report.
::testing::AssertionResult AllClose(const std::vector<float>& actual, const std::vector<double>& expected,
                                    const Tolerance& tol = kF32Default);
::testing::AssertionResult AllClose(const std::vector<float>& actual, const std::vector<double>& expected,
                                    const std::vector<double>& scale, const Tolerance& tol);

}  // namespace cudl::test
