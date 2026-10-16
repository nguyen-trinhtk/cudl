#include "cudl/core/allocator.h"

#include <cuda_runtime.h>

#include <new>

#include "cudl/core/error.h"

namespace cudl {
namespace {

constexpr std::align_val_t kCpuAlignment{64};

class CpuAllocator final : public Allocator {
public:
    void* allocate(size_t nbytes) override { return nbytes ? ::operator new(nbytes, kCpuAlignment) : nullptr; }
    void deallocate(void* ptr) noexcept override {
        if (ptr) ::operator delete(ptr, kCpuAlignment);
    }
    const char* name() const override { return "cpu"; }
};

class CudaAllocator final : public Allocator {
public:
    void* allocate(size_t nbytes) override {
        if (!nbytes) return nullptr;
        void* ptr = nullptr;
        CUDA_CHECK(cudaMalloc(&ptr, nbytes));
        return ptr;
    }
    // Errors are ignored: this can run during process teardown after the CUDA runtime is gone.
    void deallocate(void* ptr) noexcept override {
        if (ptr) cudaFree(ptr);
    }
    const char* name() const override { return "cuda"; }
};

Allocator*& slot(Device device) {
    static CpuAllocator cpu;
    static CudaAllocator cuda;
    static Allocator* slots[] = {&cpu, &cuda};
    return slots[static_cast<int>(device)];
}

}  // namespace

Allocator* get_allocator(Device device) { return slot(device); }

void set_allocator(Device device, Allocator* allocator) {
    CUDL_CHECK(allocator != nullptr, "allocator must not be null");
    slot(device) = allocator;
}

}  // namespace cudl
