add_library(serving_core
  core/dtypes.cpp
  core/tensor.cpp
  utils/tensor_index.cpp
  ops/ops.cpp
  ops/rope.cpp
  model/llama/attention.cpp
  model/llama/transformer_block.cpp
)

target_include_directories(serving_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
