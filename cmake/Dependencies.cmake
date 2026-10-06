include(FetchContent)

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
  set(_SERVING_FC_DOWNLOAD_TS DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
else()
  set(_SERVING_FC_DOWNLOAD_TS)
endif()

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
  ${_SERVING_FC_DOWNLOAD_TS}
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
  benchmark
  URL https://github.com/google/benchmark/archive/refs/tags/v1.8.3.zip
  ${_SERVING_FC_DOWNLOAD_TS}
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)
