#include "core/dtypes.hpp"
#include "core/tensor.hpp"

#include <gtest/gtest.h>

TEST(TensorTest, ZerosAndShape) {
  auto t = serving::Tensor::zeros({2, 3});
  EXPECT_EQ(t.numel(), 6u);
  EXPECT_FLOAT_EQ(t.at({0, 0}), 0.f);
  EXPECT_TRUE(t.is_contiguous());
}

TEST(TensorTest, FromVector) {
  auto t = serving::Tensor::from_vector({2, 2}, {1, 2, 3, 4});
  EXPECT_FLOAT_EQ(t.at({1, 1}), 4.f);
}

TEST(DtypeTest, Float32Size) {
  EXPECT_EQ(serving::dtype_size(serving::DType::Float32), 4u);
}
