#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace cudl::nn {
// Reproducible seeded RNG (splitmix64)
class Rng {
public:
    explicit Rng(uint64_t seed) : state_(seed) {}

    uint64_t next() {
        uint64_t z = (state_ += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }

    // [lo, hi) with 24 bits of resolution
    float uniform(float lo, float hi) {
        const float u = static_cast<float>(next() >> 40) * (1.0f / 16777216.0f);
        return lo + (hi - lo) * u;
    }

    // [0, n)
    // modulo bias is negligible for n <<<< 2^64
    uint64_t below(uint64_t n) { return next() % n; }

    // Fisher-Yates
    template <typename T>
    void shuffle(std::vector<T>& v) {
        for (size_t i = v.size(); i > 1; --i) {
            const size_t j = static_cast<size_t>(below(i));
            std::swap(v[i - 1], v[j]);
        }
    }

private:
    uint64_t state_;
};

// TODO: review distributions
// n values uniform in [-bound, bound)
inline std::vector<float> uniform_init(size_t n, float bound, Rng& rng) {
    std::vector<float> out(n);
    for (float& x : out) {
        x = rng.uniform(-bound, bound);
    }
    return out;
}

// He-Kaiming uniform (for layers followed by ReLU)
// bound = sqrt(6 / fan_in)
inline std::vector<float> he_kaiming_uniform(size_t n, size_t fan_in, Rng& rng) {
    const double bound = std::sqrt(6.0 / static_cast<double>(fan_in));
    return uniform_init(n, static_cast<float>(bound), rng);
}

// Glorot-Xavier uniform
// bound = sqrt(6 / (fan_in + fan_out))
inline std::vector<float> glorot_xavier_uniform(size_t n, size_t fan_in, size_t fan_out, Rng& rng) {
    const double bound = std::sqrt(6.0 / static_cast<double>(fan_in + fan_out));
    return uniform_init(n, static_cast<float>(bound), rng);
}
}  // namespace cudl::nn
