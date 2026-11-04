#include "common/compare.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>

namespace cudl::test {
// Helpers
namespace {
// 
int64_t ordered_bits(float f) {
    int32_t i;
    std::memcpy(&i, &f, sizeof i);
    // Let m be the exponent + mantissa bits (low 31 bits)
    // If negative
    // - Then i = INT32_MIN + m
    // - So -m = INT32_MIN - i
    // If positive
    // - Then i == m, so return i
    return (i < 0) ? 
            static_cast<int64_t>(std::numeric_limits<int32_t>::min()) - i 
            : i;
}

// how many floats are in between a and b
int64_t ulp_distance(float a, float b) {
    const int64_t d = ordered_bits(a) - ordered_bits(b);
    return d < 0 ? -d : d;
}
}  // namespace

// Summary
std::string CompareReport::summary() const {
    std::ostringstream os;
    os.precision(9);
    os << "n=" << n << " mismatches=" << mismatches << " max_abs=" << max_abs << " max_rel=" << max_rel
       << " max_ulp=" << max_ulp;
    if (n > 0) 
        os << " worst[" << worst_index << "]: actual=" << worst_actual << " expected=" << worst_expected;
    return os.str();
}

// Compare actual FP32 to expected FP64
CompareReport compare(const float* actual, const double* expected, size_t n, const Tolerance& tol,
                      const double* scale) {
    // Constants
    constexpr double kInf = std::numeric_limits<double>::infinity();
    constexpr int64_t kMaxUlp = std::numeric_limits<int64_t>::max();

    CompareReport r;
    r.n = n;
    for (size_t i = 0; i < n; ++i) {
        const float a = actual[i];
        const double e = expected[i];
        double abs_err, rel_err;
        int64_t ulp;
        bool match;

        if (std::isnan(a) || std::isnan(e)) {
            // NaNs handling
            match = std::isnan(a) && std::isnan(e);
            abs_err = rel_err = match ? 0.0 : kInf;
            ulp = match ? 0 : kMaxUlp;
        } else if (std::isinf(a) || std::isinf(e)) {
            // Inf handling
            match = static_cast<double>(a) == e;
            abs_err = rel_err = match ? 0.0 : kInf;
            ulp = match ? 0 : kMaxUlp;
        } else {
            // Regular numbers
            const double mag = scale ? scale[i] : std::fabs(e);
            abs_err = std::fabs(static_cast<double>(a) - e); 
            rel_err = mag != 0.0 ? abs_err / mag : (abs_err == 0.0 ? 0.0 : kInf);
            ulp = ulp_distance(a, static_cast<float>(e));
            match = abs_err <= tol.atol + tol.rtol * mag || (tol.max_ulp >= 0 && ulp <= tol.max_ulp);
        }

        if (!match) 
            ++r.mismatches;
        if (abs_err > r.max_abs || i == 0) {
            // Report worst index based on absolute error
            r.max_abs = abs_err;
            r.worst_index = i;
            r.worst_actual = a;
            r.worst_expected = e;
        }
        if (rel_err > r.max_rel) 
            r.max_rel = rel_err;
        if (ulp > r.max_ulp) 
            r.max_ulp = ulp;
    }
    return r;
}

// 
namespace {
::testing::AssertionResult all_close(const std::vector<float>& actual, const std::vector<double>& expected,
                                     const double* scale, const Tolerance& tol) {
    const CompareReport r = compare(actual.data(), expected.data(), actual.size(), tol, scale);
    if (r.ok()) 
        return ::testing::AssertionSuccess() << r.summary();
    return ::testing::AssertionFailure() << r.summary() << " (atol=" << tol.atol << " rtol=" << tol.rtol
                                         << " max_ulp=" << tol.max_ulp << ")";
}
}  // namespace

::testing::AssertionResult AllClose(const std::vector<float>& actual, const std::vector<double>& expected,
                                    const Tolerance& tol) {
    if (actual.size() != expected.size()) {
        return ::testing::AssertionFailure()
               << "size mismatch: actual " << actual.size() << " vs expected " << expected.size();
    }
    return all_close(actual, expected, nullptr, tol);
}

::testing::AssertionResult AllClose(const std::vector<float>& actual, const std::vector<double>& expected,
                                    const std::vector<double>& scale, const Tolerance& tol) {
    if (actual.size() != expected.size() || scale.size() != expected.size()) {
        return ::testing::AssertionFailure() << "size mismatch: actual " << actual.size() << " vs expected "
                                             << expected.size() << " vs scale " << scale.size();
    }
    return all_close(actual, expected, scale.data(), tol);
}

}  // namespace cudl::test
