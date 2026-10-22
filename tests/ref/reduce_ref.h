#pragma once

// CPU references for reductions. sum accumulates in double; max follows std::fmax (NaNs ignored).
// sum_f32_sequential / sum_f32_pairwise are fp32 baselines for the drift test, not references.

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "cudl/core/error.h"

namespace cudl::ref {

inline double reduce(const std::string& op, const float* in, size_t n) {
    if (op == "sum") {
        double acc = 0.0;
        for (size_t i = 0; i < n; ++i) acc += in[i];
        return acc;
    }
    if (op == "max") {
        double acc = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < n; ++i) acc = std::fmax(acc, static_cast<double>(in[i]));
        return acc;
    }
    CUDL_FAIL("no reference for reduce op '" << op << "'");
}

inline std::vector<double> reduce_all(const std::string& op, const std::vector<float>& in) {
    return {reduce(op, in.data(), in.size())};
}

// in is row-major [rows, cols]; reduce into one result per row.
inline std::vector<double> reduce_rows(const std::string& op, const std::vector<float>& in, size_t rows, size_t cols) {
    std::vector<double> out(rows);
    for (size_t r = 0; r < rows; ++r) out[r] = reduce(op, in.data() + r * cols, cols);
    return out;
}

// in is row-major [rows, cols]; reduce into one result per column.
inline std::vector<double> reduce_cols(const std::string& op, const std::vector<float>& in, size_t rows, size_t cols) {
    std::vector<double> out(cols);
    std::vector<float> column(rows);
    for (size_t c = 0; c < cols; ++c) {
        for (size_t r = 0; r < rows; ++r) {
            column[r] = in[r * cols + c];
        }
        out[c] = reduce(op, column.data(), rows);
    }
    return out;
}

// Left-to-right fp32 sum: error grows roughly linearly with n.
inline float sum_f32_sequential(const std::vector<float>& in) {
    float acc = 0.0f;
    for (float x : in) acc += x;
    return acc;
}

// Recursive pairwise fp32 sum: error grows with log(n).
inline float sum_f32_pairwise(const float* in, size_t n) {
    if (n <= 8) {
        float acc = 0.0f;
        for (size_t i = 0; i < n; ++i) acc += in[i];
        return acc;
    }
    const size_t half = n / 2;
    return sum_f32_pairwise(in, half) + sum_f32_pairwise(in + half, n - half);
}

}  // namespace cudl::ref
