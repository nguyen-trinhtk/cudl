#pragma once

// Device buffer whose body sits `offset` elements into the allocation, with sentinel padding on
// both sides. A non-zero offset breaks 16-byte alignment (exercising vectorized fallbacks), and
// the padding catches writes outside [0, n).

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "cudl/core/tensor.h"

namespace cudl::test {

class Guarded {
public:
    static constexpr float kSentinel = -12345.0f;
    static constexpr size_t kGuard = 7;

    Guarded(const std::vector<float>& body, size_t offset) : offset_(offset), n_(body.size()) {
        std::vector<float> host(offset + n_ + kGuard, kSentinel);
        std::copy(body.begin(), body.end(), host.begin() + offset);
        t_ = Tensor::from_vector(host, {static_cast<int64_t>(host.size())});
    }
    Guarded(size_t n, size_t offset) : Guarded(std::vector<float>(n, kSentinel), offset) {}

    float* data() { return t_.data<float>() + offset_; }

    std::vector<float> body() const {
        const auto all = t_.to_vector();
        return {all.begin() + offset_, all.begin() + offset_ + n_};
    }

    ::testing::AssertionResult guards_intact() const {
        const auto all = t_.to_vector();
        for (size_t i = 0; i < all.size(); ++i) {
            if (i >= offset_ && i < offset_ + n_) continue;
            if (all[i] != kSentinel)
                return ::testing::AssertionFailure() << "wrote outside the body at padded index " << i;
        }
        return ::testing::AssertionSuccess();
    }

private:
    size_t offset_;
    size_t n_;
    Tensor t_;
};

}  // namespace cudl::test
