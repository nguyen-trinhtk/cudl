#pragma once

// CPU reference for the fused softmax cross-entropy, computed in double.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace cudl::ref {

struct CrossEntropyResult {
    std::vector<double> loss;     // [rows]
    std::vector<double> dlogits;  // [rows, cols]
};

inline CrossEntropyResult cross_entropy_loss(const std::vector<float>& logits, const std::vector<int32_t>& labels,
                                             size_t rows, size_t cols, float grad_scale) {
    constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
    CrossEntropyResult r{std::vector<double>(rows), std::vector<double>(rows * cols)};
    for (size_t i = 0; i < rows; ++i) {
        const float* x = logits.data() + i * cols;
        double m = -std::numeric_limits<double>::infinity();
        for (size_t c = 0; c < cols; ++c) {
            m = std::max(m, static_cast<double>(x[c]));
        }
        double s = 0.0;
        for (size_t c = 0; c < cols; ++c) {
            s += std::exp(x[c] - m);
        }
        const double lse = m + std::log(s);

        const int32_t label = labels[i];
        const bool valid = label >= 0 && static_cast<size_t>(label) < cols;
        r.loss[i] = valid ? lse - x[label] : kNaN;
        for (size_t c = 0; c < cols; ++c) {
            const double p = std::exp(x[c] - lse);
            const double target = static_cast<size_t>(label) == c ? 1.0 : 0.0;
            r.dlogits[i * cols + c] = valid ? (p - target) * grad_scale : kNaN;
        }
    }
    return r;
}

}  // namespace cudl::ref
