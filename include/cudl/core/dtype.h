#pragma once

#include <cstddef>

namespace cudl {

// ========== DTYPE ===================
enum class DType { F32 };  // TODO: FP16, BF16, TF32, INT8???
// Byte size
inline size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::F32:
            return 4;
    }
    return 0;
}

// Dtype name in string
inline const char* to_string(DType dtype) {
    switch (dtype) {
        case DType::F32:
            return "f32";
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
