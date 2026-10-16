#include "cudl/core/storage.h"

namespace cudl {

Storage::Storage(size_t nbytes, Device device)
    : allocator_(get_allocator(device)), nbytes_(nbytes), device_(device), ptr_(allocator_->allocate(nbytes)) {}

Storage::~Storage() { allocator_->deallocate(ptr_); }

}  // namespace cudl
