#include "cudl/core/tensor.h"

#include <cuda_runtime.h>

#include <cstring>

#include "cudl/core/context.h"

namespace cudl {
namespace {

Shape contiguous_strides(const Shape& shape) {
    Shape strides(shape.size());
    int64_t acc = 1;
    for (size_t i = shape.size(); i-- > 0;) {
        strides[i] = acc;
        acc *= shape[i];
    }
    return strides;
}

void copy_bytes(void* dst, Device dst_device, const void* src, Device src_device, size_t nbytes) {
    if (nbytes == 0) return;
    if (dst_device == Device::CPU && src_device == Device::CPU) {
        std::memcpy(dst, src, nbytes);
        return;
    }
    cudaMemcpyKind kind = src_device == Device::CPU   ? cudaMemcpyHostToDevice
                          : dst_device == Device::CPU ? cudaMemcpyDeviceToHost
                                                      : cudaMemcpyDeviceToDevice;
    const Context& ctx = Context::get();
    CUDA_CHECK(cudaMemcpyAsync(dst, src, nbytes, kind, ctx.stream()));
    ctx.synchronize();
}

}  // namespace

Tensor Tensor::empty(Shape shape, Device device, DType dtype) {
    for (int64_t d : shape) CUDL_CHECK(d >= 0, "negative dimension " << d);
    Tensor t;
    t.shape_ = std::move(shape);
    t.strides_ = contiguous_strides(t.shape_);
    t.dtype_ = dtype;
    t.device_ = device;
    t.storage_ = std::make_shared<Storage>(t.nbytes(), device);
    return t;
}

Tensor Tensor::from_vector(const std::vector<float>& data, Shape shape, Device device) {
    Tensor t = empty(std::move(shape), device, DType::F32);
    CUDL_CHECK(static_cast<size_t>(t.numel()) == data.size(),
               "shape has " << t.numel() << " elements but data has " << data.size());
    copy_bytes(t.raw_data(), device, data.data(), Device::CPU, t.nbytes());
    return t;
}

int64_t Tensor::numel() const {
    int64_t n = 1;
    for (int64_t d : shape_) n *= d;
    return n;
}

bool Tensor::is_contiguous() const {
    int64_t expected = 1;
    for (size_t i = shape_.size(); i-- > 0;) {
        if (shape_[i] != 1 && strides_[i] != expected) return false;
        expected *= shape_[i];
    }
    return true;
}

void* Tensor::raw_data() {
    if (!storage_) return nullptr;
    return static_cast<char*>(storage_->data()) + offset_ * dtype_size(dtype_);
}

const void* Tensor::raw_data() const {
    if (!storage_) return nullptr;
    return static_cast<const char*>(storage_->data()) + offset_ * dtype_size(dtype_);
}

Tensor Tensor::to(Device device) const {
    CUDL_CHECK(defined(), "tensor is undefined");
    CUDL_CHECK(is_contiguous(), "to() requires a contiguous tensor");
    Tensor out = empty(shape_, device, dtype_);
    copy_bytes(out.raw_data(), device, raw_data(), device_, nbytes());
    return out;
}

std::vector<float> Tensor::to_vector() const {
    CUDL_CHECK(defined(), "tensor is undefined");
    CUDL_CHECK(is_contiguous(), "to_vector() requires a contiguous tensor");
    std::vector<float> out(static_cast<size_t>(numel()));
    copy_bytes(out.data(), Device::CPU, data<float>(), device_, nbytes());
    return out;
}

}  // namespace cudl
