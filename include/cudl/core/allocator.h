#pragma once

#include <cstddef>

#include "cudl/core/dtype.h"

namespace cudl {

// Wrapper class for device-agnostic memory allocation

// Swappable per device so a caching allocator can replace raw cudaMalloc later
// without touching Tensor or Storage.
class Allocator {
public:
    virtual ~Allocator() = default;
    // Returns nullptr for nbytes == 0.
    virtual void* allocate(size_t nbytes) = 0;
    virtual void deallocate(void* ptr) noexcept = 0;
    virtual const char* name() const = 0;
};

Allocator* get_allocator(Device device);
// Does not take ownership. Storage remembers the allocator it was created with,
// so swapping is safe while tensors are alive.
void set_allocator(Device device, Allocator* allocator);

}  // namespace cudl
