add_library(serving_core
  core/dtypes.cpp
  core/tensor.cpp
  utils/tensor_index.cpp
  ops/ops.cpp
)

target_include_directories(serving_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
