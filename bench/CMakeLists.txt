add_library(cudl_bench STATIC harness.cpp)
target_include_directories(cudl_bench PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cudl_bench PUBLIC cudl)
# Captured at configure time; override at run time with env CUDL_GIT_SHA.
target_compile_definitions(cudl_bench PRIVATE CUDL_GIT_SHA="${CUDL_GIT_SHA}")

foreach(name copy elementwise reduce)
    add_executable(bench_${name} bench_${name}.cpp)
    target_link_libraries(bench_${name} PRIVATE cudl_bench)
endforeach()
