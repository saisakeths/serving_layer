#include "ops/ops.hpp"

#include <cmath>
#include <gtest/gtest.h>

using namespace std;

namespace serving {
namespace {

// rms_of_row: float — root mean square of one row in a [1,dim] tensor.
float rms_of_row(const Tensor& row) {
  int64_t dim = row.shape()[1];
  float sum_sq = 0.f;
  for (int64_t d = 0; d < dim; ++d) {
    float v = row.at({0, d});
    sum_sq += v * v;
  }
  return sqrt(sum_sq / static_cast<float>(dim));
}

}  // namespace
}  // namespace serving

TEST(OpsTest, Matmul2x2) {
  auto a = serving::Tensor::from_vector({2, 2}, {1, 2, 3, 4});
  auto b = serving::Tensor::from_vector({2, 2}, {5, 6, 7, 8});
  serving::Tensor out({2, 2});
  serving::matmul(a, b, out);
  EXPECT_FLOAT_EQ(out.at({0, 0}), 19.f);
  EXPECT_FLOAT_EQ(out.at({0, 1}), 22.f);
  EXPECT_FLOAT_EQ(out.at({1, 0}), 43.f);
  EXPECT_FLOAT_EQ(out.at({1, 1}), 50.f);
}

TEST(OpsTest, RmsNormUnitRms) {
  auto x = serving::Tensor::from_vector({1, 4}, {1.f, 2.f, 3.f, 4.f});
  auto w = serving::Tensor::zeros({4});
  for (int i = 0; i < 4; ++i) {
    w.at({i}) = 1.f;
  }
  serving::Tensor out({1, 4});
  serving::rmsnorm(x, w, 1e-5f, out);
  EXPECT_NEAR(serving::rms_of_row(out), 1.f, 1e-3f);
}

TEST(OpsTest, SoftmaxSumsToOne) {
  auto x = serving::Tensor::from_vector({1, 3}, {1.f, 2.f, 3.f});
  serving::Tensor out({1, 3});
  serving::softmax_last_dim(x, out);
  float s = out.at({0, 0}) + out.at({0, 1}) + out.at({0, 2});
  EXPECT_NEAR(s, 1.f, 1e-5f);
}

TEST(OpsTest, SiluZero) {
  auto x = serving::Tensor::from_vector({1, 1}, {0.f});
  serving::silu_inplace(x);
  EXPECT_FLOAT_EQ(x.at({0, 0}), 0.f);
}

TEST(OpsTest, EmbeddingLookup) {
  auto weight = serving::Tensor::from_vector({3, 2}, {1, 2, 3, 4, 5, 6});
  vector<int32_t> ids = {0, 2};
  serving::Tensor out({2, 2});
  serving::embedding(weight, ids, out);
  EXPECT_FLOAT_EQ(out.at({0, 0}), 1.f);
  EXPECT_FLOAT_EQ(out.at({0, 1}), 2.f);
  EXPECT_FLOAT_EQ(out.at({1, 0}), 5.f);
  EXPECT_FLOAT_EQ(out.at({1, 1}), 6.f);
}

TEST(OpsTest, AddMulInplace) {
  auto a = serving::Tensor::from_vector({2}, {1.f, 2.f});
  auto b = serving::Tensor::from_vector({2}, {3.f, 4.f});
  serving::add_inplace(a, b);
  EXPECT_FLOAT_EQ(a.at({0}), 4.f);
  serving::mul_inplace(a, b);
  EXPECT_FLOAT_EQ(a.at({1}), 24.f);
}
