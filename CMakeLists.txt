cmake_minimum_required(VERSION 3.24)
project(gemm_compare C CXX)

set(CMAKE_CXX_STANDARD 11)
find_package(MKL CONFIG REQUIRED)
add_executable(gemm_compare main.cpp)
target_compile_options(gemm_compare PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
target_include_directories(gemm_compare PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(gemm_compare PUBLIC $<LINK_ONLY:MKL::MKL>)
