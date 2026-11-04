#pragma once

// CPU reference for GEMM, accumulated in double.

#include <cmath>
#include <vector>

namespace cudl::ref {

// Row-major: op(a) is [m, k], op(b) is [k, n], c is [m, n]; returns alpha * op(a) * op(b) + beta * c
// op(x) = x^T when its trans flag is set: a is then stored [k, m] and b [n, k]
// beta == 0 ignores c (BLAS semantics), so NaNs in c don't propagate
inline std::vector<double> gemm(const std::vector<float>& a, const std::vector<float>& b, const std::vector<float>& c,
                                size_t m, size_t n, size_t k, bool trans_a, bool trans_b, float alpha, float beta) {
    std::vector<double> ab(m * n, 0.0);
    for (size_t r = 0; r < m; ++r) {
        for (size_t i = 0; i < k; ++i) {
            const double av = trans_a ? a[i * m + r] : a[r * k + i]; //transposed or not
            for (size_t col = 0; col < n; ++col) {
                const double bv = trans_b ? b[col * k + i] : b[i * n + col]; //tranposed or not
                ab[r * n + col] += av * bv;
            }
        }
    }
    std::vector<double> out(m * n);
    for (size_t i = 0; i < m * n; ++i) {
        const double scaled = static_cast<double>(alpha) * ab[i];
        const double prev = (beta == 0.0f) ? 0.0 : (static_cast<double>(beta) * c[i]);
        out[i] = scaled + prev;
    }
    return out;
}

// Magnitude for the fp32 rounding-error bound of alpha * op(a) * op(b) + beta * c:
//   scale = |alpha| * |op(a)| * |op(b)| + |beta| * |c|     (elementwise abs, then the same gemm)
// The kernel accumulates k products in fp32, then rounds alpha * acc, beta * c and their sum, so for any summation
// order |fl(out) - out| <= gamma_{k+2} * scale (gamma_n = n u / (1 - n u), u = 2^-24). Pass as `scale` to
// test::compare with rtol = gamma_{k+2}; unlike |out| it doesn't vanish when the products cancel.
inline std::vector<double> gemm_error_scale(const std::vector<float>& a, const std::vector<float>& b,
                                            const std::vector<float>& c, size_t m, size_t n, size_t k, bool trans_a,
                                            bool trans_b, float alpha, float beta) {
    const auto abs = [](const std::vector<float>& x) {
        std::vector<float> out(x.size());
        for (size_t i = 0; i < x.size(); ++i) out[i] = std::fabs(x[i]);
        return out;
    };
    return gemm(abs(a), abs(b), abs(c), m, n, k, trans_a, trans_b, std::fabs(alpha), std::fabs(beta));
}

// gamma_n = n u / (1 - n u) with u = 2^-24 (fp32 unit roundoff): bound on the relative error of n roundings
inline double gamma_f32(size_t n) {
    const double nu = static_cast<double>(n) * std::ldexp(1.0, -24);
    return nu / (1.0 - nu);
}
}  // namespace cudl::ref
