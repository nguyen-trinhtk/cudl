#pragma once

// CPU reference MLP in double

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace cudl::ref {

// y = x W + b with W stored [in, out] to match nn::Linear
struct LinearRef {
    size_t in = 0, out = 0;
    std::vector<double> w;  // [in, out]
    std::vector<double> b;  // [out]
};

struct LinearGrads {
    std::vector<double> dw, db;
};

// x [batch, in] -> [batch, out]
inline std::vector<double> linear_forward(const LinearRef& l, const std::vector<double>& x, size_t batch) {
    std::vector<double> y(batch * l.out);
    for (size_t r = 0; r < batch; ++r) {
        for (size_t o = 0; o < l.out; ++o) {
            y[r * l.out + o] = l.b[o];
        }
        for (size_t i = 0; i < l.in; ++i) {
            const double xv = x[r * l.in + i];
            for (size_t o = 0; o < l.out; ++o) {
                y[r * l.out + o] += xv * l.w[i * l.out + o];
            }
        }
    }
    return y;
}

// Writes dW = x^T dy and db = sum_rows(dy)
// returns dx = dy W^T
inline std::vector<double> linear_backward(const LinearRef& l, const std::vector<double>& x,
                                           const std::vector<double>& dy, size_t batch, LinearGrads& g) {
    g.dw.assign(l.in * l.out, 0.0);
    g.db.assign(l.out, 0.0);
    std::vector<double> dx(batch * l.in, 0.0);
    for (size_t r = 0; r < batch; ++r) {
        for (size_t o = 0; o < l.out; ++o) {
            g.db[o] += dy[r * l.out + o];
        }
        for (size_t i = 0; i < l.in; ++i) {
            const double xv = x[r * l.in + i];
            double acc = 0.0;
            for (size_t o = 0; o < l.out; ++o) {
                const double d = dy[r * l.out + o];
                g.dw[i * l.out + o] += xv * d;
                acc += d * l.w[i * l.out + o];
            }
            dx[r * l.in + i] = acc;
        }
    }
    return dx;
}

// Mean softmax cross-entropy of logits [batch, classes]; writes dL/dlogits when dlogits != nullptr
inline double mean_cross_entropy(const std::vector<double>& logits, const std::vector<int32_t>& labels, size_t batch,
                        size_t classes, std::vector<double>* dlogits) {
    if (dlogits) {
        dlogits->assign(batch * classes, 0.0);
    }
    double total = 0.0;
    for (size_t r = 0; r < batch; ++r) {
        const double* x = logits.data() + r * classes;
        double m = -std::numeric_limits<double>::infinity();
        for (size_t c = 0; c < classes; ++c) {
            m = std::max(m, x[c]);
        }
        double s = 0.0;
        for (size_t c = 0; c < classes; ++c) {
            s += std::exp(x[c] - m);
        }
        const double lse = m + std::log(s);
        total += lse - x[labels[r]];
        if (dlogits) {
            for (size_t c = 0; c < classes; ++c) {
                const double p = std::exp(x[c] - lse);
                const double target = static_cast<size_t>(labels[r]) == c ? 1.0 : 0.0;
                (*dlogits)[r * classes + c] = (p - target) / static_cast<double>(batch);
            }
        }
    }
    return total / static_cast<double>(batch);
}

struct MlpGrads {
    std::vector<LinearGrads> layers;
    std::vector<double> dx;  // dL/d(input)
};

// Loss of the MLP on one batch
// fills grads when grads != nullptr
inline double mlp_loss(const std::vector<LinearRef>& layers, const std::vector<double>& x, size_t batch,
                       const std::vector<int32_t>& labels, MlpGrads* grads = nullptr) {
    // acts[i] is the input to layer i
    // pre[i] its output before the ReLU
    std::vector<std::vector<double>> acts{x}, pre;
    for (size_t i = 0; i < layers.size(); ++i) {
        pre.push_back(linear_forward(layers[i], acts.back(), batch));
        if (i + 1 < layers.size()) {
            std::vector<double> h = pre.back();
            for (double& v : h) {
                v = v > 0.0 ? v : 0.0;
            }
            acts.push_back(std::move(h));
        }
    }
    const size_t classes = layers.back().out;
    std::vector<double> dy;
    const double loss = mean_cross_entropy(pre.back(), labels, batch, classes, grads ? &dy : nullptr);
    if (!grads) return loss;

    grads->layers.assign(layers.size(), {});
    for (size_t i = layers.size(); i-- > 0;) {
        if (i + 1 < layers.size()) {
            for (size_t j = 0; j < dy.size(); ++j) {
                dy[j] = pre[i][j] > 0.0 ? dy[j] : 0.0;
            }
        }
        dy = linear_backward(layers[i], acts[i], dy, batch, grads->layers[i]);
    }
    grads->dx = std::move(dy);
    return loss;
}

}  // namespace cudl::ref
