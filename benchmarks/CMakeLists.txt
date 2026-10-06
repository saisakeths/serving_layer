add_executable(bench_ops bench_ops.cpp)
target_link_libraries(bench_ops PRIVATE serving_core benchmark::benchmark)

add_executable(bench_attention bench_attention.cpp)
target_link_libraries(bench_attention PRIVATE serving_core benchmark::benchmark)
