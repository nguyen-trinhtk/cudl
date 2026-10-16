// Tensor basics: shapes/strides, zero-sized tensors, host<->device round trips,
// and argument validation.

#include <gtest/gtest.h>

#include "common/random.h"
#include "cudl/core/tensor.h"

namespace cudl {
namespace {

TEST(Tensor, ContiguousStrides) {
    Tensor t = Tensor::empty({2, 3, 4});
    EXPECT_EQ(t.strides(), (Shape{12, 4, 1}));
    EXPECT_EQ(t.numel(), 24);
    EXPECT_EQ(t.nbytes(), 96u);
    EXPECT_TRUE(t.is_contiguous());
    EXPECT_EQ(t.device(), Device::CUDA);
    EXPECT_NE(t.data<float>(), nullptr);
}

TEST(Tensor, ZeroSized) {
    Tensor t = Tensor::empty({0, 5});
    EXPECT_EQ(t.numel(), 0);
    EXPECT_TRUE(t.to_vector().empty());
}

TEST(Tensor, HostDeviceRoundTrip) {
    const auto host = test::random_uniform(1000, /*seed=*/7);
    Tensor d = Tensor::from_vector(host, {10, 100});
    EXPECT_EQ(d.to_vector(), host);
}

TEST(Tensor, ToCpuAndBack) {
    const auto host = test::random_uniform(257, /*seed=*/11);
    Tensor h = Tensor::from_vector(host, {257}).to(Device::CPU);
    EXPECT_EQ(h.device(), Device::CPU);
    EXPECT_EQ(h.to_vector(), host);
    EXPECT_EQ(h.to(Device::CUDA).to_vector(), host);
}

TEST(Tensor, FromVectorSizeMismatchThrows) {
    EXPECT_THROW(Tensor::from_vector(std::vector<float>(5), {2, 3}), std::runtime_error);
}

TEST(Tensor, NegativeDimThrows) { EXPECT_THROW(Tensor::empty({2, -1}), std::runtime_error); }

}  // namespace
}  // namespace cudl
