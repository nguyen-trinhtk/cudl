#include <mutex>

#include "cudl/kernels/copy.h"
#include "cudl/kernels/elementwise.h"
#include "cudl/kernels/gemm.h"
#include "cudl/kernels/reduce.h"
#include "cudl/kernels/registry.h"

namespace cudl::kernels::detail {

// Add one line per kernel family.
void ensure_kernels_registered() {
    static std::once_flag once;
    std::call_once(once, [] {
        register_copy_kernels();
        register_elementwise_kernels();
        register_reduce_kernels();
        register_gemm_kernels();
    });
}
}  // namespace cudl::kernels::detail
