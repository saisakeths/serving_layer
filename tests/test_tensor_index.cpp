#include "utils/tensor_index.hpp"

#include <gtest/gtest.h>

using namespace std;

namespace serving {
namespace {

// row_major_strides: vector<int64_t> — builds row-major strides for the given shape.
vector<int64_t> row_major_strides(const vector<int64_t>& shape) {
  vector<int64_t> strides(shape.size(), 1);
  if (shape.size() <= 1) {
    return strides;
  }
  for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

}  // namespace
}  // namespace serving

TEST(TensorIndexTest, LinearIndex2x3) {
  vector<int64_t> shape = {2, 3};
  auto strides = serving::row_major_strides(shape);
  EXPECT_EQ(serving::TensorIndex::linear_index(shape, strides, {0, 0}), 0);
  EXPECT_EQ(serving::TensorIndex::linear_index(shape, strides, {0, 2}), 2);
  EXPECT_EQ(serving::TensorIndex::linear_index(shape, strides, {1, 0}), 3);
  EXPECT_EQ(serving::TensorIndex::linear_index(shape, strides, {1, 2}), 5);
}

TEST(TensorIndexTest, RankMismatchThrows) {
  vector<int64_t> shape = {2, 3};
  auto strides = serving::row_major_strides(shape);
  EXPECT_THROW(serving::TensorIndex::linear_index(shape, strides, {0}), invalid_argument);
}

TEST(TensorIndexTest, OutOfRangeThrows) {
  vector<int64_t> shape = {2, 3};
  auto strides = serving::row_major_strides(shape);
  EXPECT_THROW(serving::TensorIndex::linear_index(shape, strides, {2, 0}), out_of_range);
}
