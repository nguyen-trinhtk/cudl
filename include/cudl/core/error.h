#pragma once

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace cudl {

// Wrapper class for CUDA errors
class CudaError : public std::runtime_error {
public:
    CudaError(cudaError_t code, const std::string& what) : std::runtime_error(what), code_(code) {}
    cudaError_t code() const { return code_; }

private:
    cudaError_t code_;
};

namespace detail {

// CUDA error
[[noreturn]] inline void throw_cuda_error(cudaError_t code, const char* expr, const char* file, int line) {
    std::ostringstream os;
    os << "CUDA error " << cudaGetErrorName(code) << " (" << cudaGetErrorString(code) << ") at " << file << ":" << line
       << ": " << expr;
    throw CudaError(code, os.str());
}

// General error
[[noreturn]] inline void throw_error(const std::string& msg, const char* file, int line) {
    std::ostringstream os;
    os << msg << " (" << file << ":" << line << ")";
    throw std::runtime_error(os.str());
}

}  // namespace detail
}  // namespace cudl

// Macro for checking for CUDA error in exec
#define CUDA_CHECK(expr)                                                                                      \
    do {                                                                                                      \
        cudaError_t cudl_err_ = (expr);                                                                       \
        if (cudl_err_ != cudaSuccess) ::cudl::detail::throw_cuda_error(cudl_err_, #expr, __FILE__, __LINE__); \
    } while (0)

// Call right after a kernel launch
// With CUDL_SYNC_KERNELS, also synchronizes so
// asynchronous faults are reported at the launch that caused them.
#ifdef CUDL_SYNC_KERNELS
#define CUDL_KERNEL_CHECK()                  \
    do {                                     \
        CUDA_CHECK(cudaGetLastError());      \
        CUDA_CHECK(cudaDeviceSynchronize()); \
    } while (0)
#else
#define CUDL_KERNEL_CHECK() CUDA_CHECK(cudaGetLastError())
#endif

// Custom check macros
// `msg` is streamed, e.g. CUDL_FAIL("bad size " << n).
#define CUDL_FAIL(msg)                                                   \
    do {                                                                 \
        std::ostringstream cudl_os_;                                     \
        cudl_os_ << msg;                                                 \
        ::cudl::detail::throw_error(cudl_os_.str(), __FILE__, __LINE__); \
    } while (0)

#define CUDL_CHECK(cond, msg)                                       \
    do {                                                            \
        if (!(cond)) \
        CUDL_FAIL("check failed: " #cond ": " << msg); \
    } while (0)
