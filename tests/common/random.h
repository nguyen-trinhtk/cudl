#pragma once

// Seeded random test data that is identical on every platform, so a failing seed
// reproduces anywhere.

#include <cstdint>
#include <vector>

namespace cudl::test {

inline uint64_t splitmix64(uint64_t& state) {
    uint64_t z = (state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// Unlike std::uniform_real_distribution, the output is identical across standard
// libraries, so a failing seed reproduces anywhere.
inline std::vector<float> random_uniform(size_t n, uint64_t seed, float lo = -1.0f, float hi = 1.0f) {
    std::vector<float> out(n);
    uint64_t state = seed;
    for (float& x : out) {
        const float u = static_cast<float>(splitmix64(state) >> 40) * (1.0f / 16777216.0f);  // [0, 1)
        x = lo + (hi - lo) * u;
    }
    return out;
}

}  // namespace cudl::test
