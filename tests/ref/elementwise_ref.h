#pragma once

// CPU references for elementwise ops, computed in double.

#include <cmath>
#include <functional>
#include <string>
#include <vector>

#include "cudl/core/error.h"

namespace cudl::ref {

using UnaryOp = std::function<double(double)>;
using BinaryOp = std::function<double(double, double)>;

inline UnaryOp unary_op(const std::string& op) {
    if (op == "neg") return [](double x) { return -x; };
    if (op == "relu") return [](double x) { return x < 0.0 ? 0.0 : x; };
    if (op == "exp") return [](double x) { return std::exp(x); };
    if (op == "tanh") return [](double x) { return std::tanh(x); };
    if (op == "sigmoid") return [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
    CUDL_FAIL("no reference for unary op '" << op << "'");
}

inline BinaryOp binary_op(const std::string& op) {
    if (op == "add") return [](double a, double b) { return a + b; };
    if (op == "sub") return [](double a, double b) { return a - b; };
    if (op == "mul") return [](double a, double b) { return a * b; };
    if (op == "div") return [](double a, double b) { return a / b; };
    CUDL_FAIL("no reference for binary op '" << op << "'");
}

inline std::vector<double> unary(const std::string& op, const std::vector<float>& in) {
    const UnaryOp f = unary_op(op);
    std::vector<double> out(in.size());
    for (size_t i = 0; i < in.size(); ++i) out[i] = f(in[i]);
    return out;
}

inline std::vector<double> scale(const std::vector<float>& in, float alpha) {
    std::vector<double> out(in.size());
    for (size_t i = 0; i < in.size(); ++i) out[i] = static_cast<double>(alpha) * in[i];
    return out;
}

inline std::vector<double> binary(const std::string& op, const std::vector<float>& a, const std::vector<float>& b) {
    const BinaryOp f = binary_op(op);
    std::vector<double> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) out[i] = f(a[i], b[i]);
    return out;
}

// x is row-major [rows, cols]; bias has cols elements.
inline std::vector<double> add_bias(const std::vector<float>& x, const std::vector<float>& bias, size_t rows,
                                    size_t cols) {
    std::vector<double> out(rows * cols);
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < cols; ++c) out[r * cols + c] = static_cast<double>(x[r * cols + c]) + bias[c];
    return out;
}

}  // namespace cudl::ref
