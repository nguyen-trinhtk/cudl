#pragma once

// CPU references for the optimizer update rules, with parameters and state held in double.

#include <cmath>
#include <vector>

namespace cudl::ref {

// v = momentum * v + g; p -= lr * v  (momentum == 0: p -= lr * g, v untouched)
inline void sgd_step(std::vector<double>& param, const std::vector<double>& grad, std::vector<double>& velocity,
                     double lr, double momentum) {
    for (size_t i = 0; i < param.size(); ++i) {
        double step = grad[i];
        if (momentum != 0.0) {
            step = momentum * velocity[i] + step;
            velocity[i] = step;
        }
        param[i] -= lr * step;
    }
}

// Adam step number t (starting at 1)
inline void adam_step(std::vector<double>& param, const std::vector<double>& grad, std::vector<double>& m,
                      std::vector<double>& v, long t, double lr, double beta1, double beta2, double eps) {
    const double bc1 = 1.0 - std::pow(beta1, static_cast<double>(t));
    const double bc2 = 1.0 - std::pow(beta2, static_cast<double>(t));
    for (size_t i = 0; i < param.size(); ++i) {
        const double g = grad[i];
        m[i] = beta1 * m[i] + (1.0 - beta1) * g;
        v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
        const double m_hat = m[i] / bc1;
        const double v_hat = v[i] / bc2;
        param[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
    }
}

}  // namespace cudl::ref
