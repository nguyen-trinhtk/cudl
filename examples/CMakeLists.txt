add_executable(mlp_mnist mlp_mnist.cpp)
target_link_libraries(mlp_mnist PRIVATE cudl)
