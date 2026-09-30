add_executable(test_tensor test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE serving_core GTest::gtest_main)

add_executable(test_tensor_index test_tensor_index.cpp)
target_link_libraries(test_tensor_index PRIVATE serving_core GTest::gtest_main)

add_executable(test_ops test_ops.cpp)
target_link_libraries(test_ops PRIVATE serving_core GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(test_tensor)
gtest_discover_tests(test_tensor_index)
gtest_discover_tests(test_ops)
