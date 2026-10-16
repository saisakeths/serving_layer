add_executable(mini_infer mini_infer/main.cpp)
target_link_libraries(mini_infer PRIVATE serving_core)
