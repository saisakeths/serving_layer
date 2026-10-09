add_executable(test_tensor test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE serving_core GTest::gtest_main)

add_executable(test_tensor_index test_tensor_index.cpp)
target_link_libraries(test_tensor_index PRIVATE serving_core GTest::gtest_main)

add_executable(test_ops test_ops.cpp)
target_link_libraries(test_ops PRIVATE serving_core GTest::gtest_main)

add_executable(test_attention test_attention.cpp)
target_link_libraries(test_attention PRIVATE serving_core GTest::gtest_main)

add_executable(test_one_block_forward test_one_block_forward.cpp)
target_link_libraries(test_one_block_forward PRIVATE serving_core GTest::gtest_main)

add_executable(test_kv_cache test_kv_cache.cpp)
target_link_libraries(test_kv_cache PRIVATE serving_core GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(test_tensor)
gtest_discover_tests(test_tensor_index)
gtest_discover_tests(test_ops)
gtest_discover_tests(test_attention)
gtest_discover_tests(test_one_block_forward)
gtest_discover_tests(test_kv_cache)
