add_library(cudl_bench STATIC harness.cpp)
target_include_directories(cudl_bench PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cudl_bench PUBLIC cudl)
# Captured at configure time; override at run time with env CUDL_GIT_SHA.
target_compile_definitions(cudl_bench PRIVATE CUDL_GIT_SHA="${CUDL_GIT_SHA}")

foreach(name copy elementwise reduce gemm nn)
    add_executable(bench_${name} bench_${name}.cpp)
    target_link_libraries(bench_${name} PRIVATE cudl_bench)
endforeach()

# Tile size sweep for the smem_2d GEMM (a .cu file since it builds the kernels itself)
add_executable(bench_gemm_tune bench_gemm_tune.cu)
target_include_directories(bench_gemm_tune PRIVATE ${PROJECT_SOURCE_DIR}/src/kernels)
target_link_libraries(bench_gemm_tune PRIVATE cudl_bench)
target_compile_options(bench_gemm_tune PRIVATE -lineinfo --expt-relaxed-constexpr)
