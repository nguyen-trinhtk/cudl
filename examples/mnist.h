#pragma once

// Minimal reader for the MNIST IDX files (http://yann.lecun.com/exdb/mnist/), uncompressed.
// scripts/get_mnist.sh downloads them.

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "cudl/core/error.h"

namespace mnist {

struct Dataset {
    std::vector<float> images;    // [n, 784], normalized with the training set's mean and std
    std::vector<int32_t> labels;  // [n], 0..9
    int64_t n = 0;
    static constexpr int64_t kFeatures = 28 * 28;
    static constexpr int64_t kClasses = 10;
};

namespace detail {

// Note: IDX headers are big-endian
inline uint32_t read_u32(std::ifstream& f) {
    unsigned char b[4];
    f.read(reinterpret_cast<char*>(b), 4);
    const uint32_t hi = (static_cast<uint32_t>(b[0]) << 24) | (static_cast<uint32_t>(b[1]) << 16);
    const uint32_t lo = (static_cast<uint32_t>(b[2]) << 8) | static_cast<uint32_t>(b[3]);
    return hi | lo;
}

inline std::ifstream open(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    CUDL_CHECK(f.good(), "cannot open " << path << " (run scripts/get_mnist.sh)");
    return f;
}

}  // namespace detail

// split is "train" or "t10k"
inline Dataset load(const std::string& dir, const std::string& split) {
    // Mean and std of the training pixels in [0, 1], the usual MNIST normalization
    constexpr float kMean = 0.1307f;
    constexpr float kStd = 0.3081f;

    Dataset d;
    std::ifstream img = detail::open(dir + "/" + split + "-images-idx3-ubyte");
    CUDL_CHECK(detail::read_u32(img) == 2051, "bad magic in " << split << " images");
    const uint32_t n = detail::read_u32(img);
    const uint32_t rows = detail::read_u32(img);
    const uint32_t cols = detail::read_u32(img);
    CUDL_CHECK(rows * cols == Dataset::kFeatures, "expected 28x28 images, got " << rows << "x" << cols);

    std::vector<unsigned char> pixels(static_cast<size_t>(n) * Dataset::kFeatures);
    img.read(reinterpret_cast<char*>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
    CUDL_CHECK(img.good(), "truncated " << split << " images");
    d.images.resize(pixels.size());
    for (size_t i = 0; i < pixels.size(); ++i) {
        const float x = static_cast<float>(pixels[i]) / 255.0f;
        d.images[i] = (x - kMean) / kStd;
    }

    std::ifstream lbl = detail::open(dir + "/" + split + "-labels-idx1-ubyte");
    CUDL_CHECK(detail::read_u32(lbl) == 2049, "bad magic in " << split << " labels");
    CUDL_CHECK(detail::read_u32(lbl) == n, split << " image and label counts differ");
    std::vector<unsigned char> raw(n);
    lbl.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
    CUDL_CHECK(lbl.good(), "truncated " << split << " labels");
    d.labels.assign(raw.begin(), raw.end());
    d.n = n;
    return d;
}

}  // namespace mnist
