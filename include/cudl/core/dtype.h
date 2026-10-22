#pragma once

#include <cstddef>
#include <cstdint>

namespace cudl {

// ========== DTYPE ===================
enum class DType { F32, I32 };  // TODO: FP16, BF16, TF32, INT8???
// For now kernels compute in F32
// I32 is for integer data (e.g. class labels)

// Byte size
inline size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::F32:
            return 4;
        case DType::I32:
            return 4;
    }
    return 0;
}

// Dtype name in string
inline const char* to_string(DType dtype) {
    switch (dtype) {
        case DType::F32:
            return "f32";
        case DType::I32:
            return "i32";
    }
    return "?";
}

// Type check utils
template <typename T>
struct DTypeOf;

template <>
struct DTypeOf<float> {
    static constexpr DType value = DType::F32;
};

template <>
struct DTypeOf<int32_t> {
    static constexpr DType value = DType::I32;
};

// ========== DEVICE ===================
enum class Device { CPU, CUDA };
inline const char* to_string(Device device) {
    switch (device) {
        case Device::CPU:
            return "cpu";
        case Device::CUDA:
            return "cuda";
    }
    return "?";
}
}  // namespace cudl
