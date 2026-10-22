#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "cudl/core/dtype.h"
#include "cudl/core/error.h"
#include "cudl/core/storage.h"

namespace cudl {

using Shape = std::vector<int64_t>;

// Analogous to C++ views
// N-d view over a Storage (raw memory)
class Tensor {
public:
    Tensor() = default;

    static Tensor empty(Shape shape, Device device = Device::CUDA, DType dtype = DType::F32);
    static Tensor zeros(Shape shape, Device device = Device::CUDA, DType dtype = DType::F32);
    static Tensor from_vector(const std::vector<float>& data, Shape shape, Device device = Device::CUDA);
    static Tensor from_vector(const std::vector<int32_t>& data, Shape shape, Device device = Device::CUDA);

    bool defined() const { return storage_ != nullptr; }
    const Shape& shape() const { return shape_; }
    const Shape& strides() const { return strides_; }
    int64_t dim() const { return static_cast<int64_t>(shape_.size()); }
    int64_t numel() const;
    size_t nbytes() const { return static_cast<size_t>(numel()) * dtype_size(dtype_); }
    DType dtype() const { return dtype_; }
    Device device() const { return device_; }
    bool is_contiguous() const;

    void* raw_data();
    const void* raw_data() const;

    template <typename T>
    T* data() {
        check_dtype<T>();
        return static_cast<T*>(raw_data());
    }
    template <typename T>
    const T* data() const {
        check_dtype<T>();
        return static_cast<const T*>(raw_data());
    }

    // Synchronous copies (slow perf)
    // Only for setup and tests
    Tensor to(Device device) const;
    std::vector<float> to_vector() const;
    // Overwrite the whole tensor from host memory
    // nbytes must equal this->nbytes()
    void copy_from_host(const void* src, size_t nbytes);

private:
    template <typename T>
    void check_dtype() const {
        CUDL_CHECK(DTypeOf<T>::value == dtype_, "tensor dtype is " << to_string(dtype_));
    }

    std::shared_ptr<Storage> storage_;
    Shape shape_;
    Shape strides_;       // units: elements
    int64_t offset_ = 0;  // units: elements
    DType dtype_ = DType::F32;
    Device device_ = Device::CUDA;
};

}  // namespace cudl
