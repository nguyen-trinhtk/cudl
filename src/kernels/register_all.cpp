#include <mutex>

#include "cudl/kernels/copy.h"
#include "cudl/kernels/registry.h"

namespace cudl::kernels::detail {

// Add one line per kernel family.
void ensure_kernels_registered() {
    static std::once_flag once;
    std::call_once(once, [] { register_copy_kernels(); });
}
}  // namespace cudl::kernels::detail
