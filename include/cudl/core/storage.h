#pragma once

#include <cstddef>

#include "cudl/core/allocator.h"
#include "cudl/core/dtype.h"

namespace cudl {

// Owns one raw block of mem alloc
// Shared between tensors via shared_ptr
class Storage {
public:
    Storage(size_t nbytes, Device device);
    ~Storage();
    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;

    void* data() { return ptr_; }
    const void* data() const { return ptr_; }
    size_t nbytes() const { return nbytes_; }
    Device device() const { return device_; }

private:
    Allocator* allocator_;
    size_t nbytes_;
    Device device_;
    void* ptr_;
};

}  // namespace cudl
