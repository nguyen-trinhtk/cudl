cmake_minimum_required(VERSION 3.24)

# Must be set before project(), otherwise CMake picks the compiler default.
# Override on GPU-less build nodes, e.g. -DCMAKE_CUDA_ARCHITECTURES=80.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(cudl LANGUAGES CXX CUDA)

option(CUDL_BUILD_TESTS "Build unit tests" ON)
option(CUDL_BUILD_BENCH "Build benchmarks" ON)
option(CUDL_BUILD_EXAMPLES "Build examples" ON)
option(CUDL_SYNC_KERNELS "Synchronize after every kernel launch (debugging)" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)

execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE CUDL_GIT_SHA
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
if(NOT CUDL_GIT_SHA)
    set(CUDL_GIT_SHA unknown)
endif()

add_library(cudl STATIC
    src/core/allocator.cpp
    src/core/context.cpp
    src/core/storage.cpp
    src/core/tensor.cpp
    src/kernels/register_all.cpp
    src/kernels/copy.cu
    src/kernels/elementwise.cu
    src/kernels/reduce.cu
    src/kernels/gemm.cu
    src/kernels/loss.cu
    src/kernels/optim.cu
    src/nn/layers.cpp
    src/nn/loss.cpp
    src/optim/optim.cpp)

target_include_directories(cudl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(cudl PUBLIC CUDA::cudart)
target_compile_options(cudl PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --expt-relaxed-constexpr -Xcompiler=-Wall>)

if(CUDL_SYNC_KERNELS)
    target_compile_definitions(cudl PUBLIC CUDL_SYNC_KERNELS)
endif()

# NVTX3 is header-only and ships with the CUDA toolkit.
if(TARGET CUDA::nvtx3)
    target_link_libraries(cudl PUBLIC CUDA::nvtx3)
else()
    find_path(CUDL_NVTX3_INCLUDE nvtx3/nvToolsExt.h HINTS ${CUDAToolkit_INCLUDE_DIRS})
    if(CUDL_NVTX3_INCLUDE)
        target_include_directories(cudl PUBLIC ${CUDL_NVTX3_INCLUDE})
        target_link_libraries(cudl PUBLIC ${CMAKE_DL_LIBS})
    else()
        message(WARNING "NVTX3 headers not found; NVTX ranges disabled")
        target_compile_definitions(cudl PUBLIC CUDL_NO_NVTX)
    endif()
endif()

if(CUDL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(CUDL_BUILD_BENCH)
    add_subdirectory(bench)
endif()

if(CUDL_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
