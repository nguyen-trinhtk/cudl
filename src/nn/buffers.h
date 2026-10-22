#pragma once

#include "cudl/core/error.h"
#include "cudl/core/tensor.h"

namespace cudl::nn::detail {

// For lazy buffer allocation (only realloc if size changes)
inline Tensor& ensure(Tensor& t, const Shape& shape) {
    if (!t.defined() || t.shape() != shape) {
        t = Tensor::empty(shape);
    }
    return t;
}

// Check: contiguous F32 RxC tensor on the GPU
inline void check_matrix(const Tensor& x, int64_t cols, const char* what) {
    CUDL_CHECK(x.defined(), what << " is undefined");
    CUDL_CHECK(x.device() == Device::CUDA, what << " must be on the GPU");
    CUDL_CHECK(x.dtype() == DType::F32, what << " must be f32, got " << to_string(x.dtype()));
    CUDL_CHECK(x.is_contiguous(), what << " must be contiguous");
    CUDL_CHECK(x.dim() == 2 && x.shape()[1] == cols, what << " must be [batch, " << cols << "]");
}
}  // namespace cudl::nn::detail
