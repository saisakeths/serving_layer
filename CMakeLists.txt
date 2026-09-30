cmake_minimum_required(VERSION 3.20)
project(serving_layer VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(SERVING_BUILD_TESTS "Build unit tests" ON)
option(SERVING_BUILD_BENCHMARKS "Build benchmarks" ON)

include(cmake/Dependencies.cmake)

add_subdirectory(src)

if(SERVING_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(SERVING_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
