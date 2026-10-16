include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Shared test utilities (comparison, random data) and CPU references in ref/.
add_library(cudl_testing STATIC common/compare.cpp)
target_include_directories(cudl_testing PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cudl_testing PUBLIC cudl GTest::gtest)

add_executable(cudl_tests
    test_tensor.cpp
    test_copy.cpp)
target_link_libraries(cudl_tests PRIVATE cudl_testing GTest::gtest_main)

include(GoogleTest)
# PRE_TEST: list tests when ctest runs, so building doesn't need a GPU.
gtest_discover_tests(cudl_tests DISCOVERY_MODE PRE_TEST)
