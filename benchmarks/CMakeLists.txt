add_executable(bench_ops bench_ops.cpp)
target_link_libraries(bench_ops PRIVATE serving_core benchmark::benchmark)
