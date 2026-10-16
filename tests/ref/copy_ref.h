#pragma once

// Naive CPU reference implementations that accumulate in double. They are the
// "truth" every GPU kernel is checked against.

#include <vector>

namespace cudl::ref {

inline std::vector<double> copy(const std::vector<float>& src) { return {src.begin(), src.end()}; }

}  // namespace cudl::ref
