#pragma once

// CPU reference for GEMM, accumulated in double.

#include <vector>

namespace cudl::ref {

// Row-major: a is [m, k], b is [k, n], c is [m, n]; returns alpha * a * b + beta * c.
// beta == 0 ignores c (BLAS semantics), so NaNs in c don't propagate.
inline std::vector<double> gemm(const std::vector<float>& a, const std::vector<float>& b, const std::vector<float>& c,
                                size_t m, size_t n, size_t k, float alpha, float beta) {
    std::vector<double> ab(m * n, 0.0);
    for (size_t r = 0; r < m; ++r) {
        for (size_t i = 0; i < k; ++i) {
            const double av = a[r * k + i];
            for (size_t col = 0; col < n; ++col) {
                ab[r * n + col] += av * b[i * n + col];
            }
        }
    }
    std::vector<double> out(m * n);
    for (size_t i = 0; i < m * n; ++i) {
        const double scaled = static_cast<double>(alpha) * ab[i];
        const double prev = beta == 0.0f ? 0.0 : static_cast<double>(beta) * c[i];
        out[i] = scaled + prev;
    }
    return out;
}

}  // namespace cudl::ref
