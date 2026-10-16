add_library(cudl_bench STATIC harness.cpp)
target_include_directories(cudl_bench PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cudl_bench PUBLIC cudl)
# Captured at configure time; override at run time with env CUDL_GIT_SHA.
target_compile_definitions(cudl_bench PRIVATE CUDL_GIT_SHA="${CUDL_GIT_SHA}")

add_executable(bench_copy bench_copy.cpp)
target_link_libraries(bench_copy PRIVATE cudl_bench)
