#include "sampler/sampler.hpp"

#include <gtest/gtest.h>

using namespace std;

namespace serving {

TEST(SamplerTest, GreedyArgmax) {
  Tensor logits({1, 1, 4});
  logits.at({0, 0, 0}) = 0.1f;
  logits.at({0, 0, 1}) = 2.5f;
  logits.at({0, 0, 2}) = 1.0f;
  logits.at({0, 0, 3}) = -1.f;
  SamplerConfig cfg;
  cfg.greedy = true;
  Sampler sampler(cfg);
  EXPECT_EQ(sampler.sample(logits), 1);
}

TEST(SamplerTest, TemperatureSeedDeterministic) {
  Tensor logits({1, 4});
  logits.at({0, 0}) = 1.f;
  logits.at({0, 1}) = 2.f;
  logits.at({0, 2}) = 3.f;
  logits.at({0, 3}) = 4.f;
  SamplerConfig cfg;
  cfg.greedy = false;
  cfg.temperature = 1.f;
  cfg.seed = 42;
  Sampler a(cfg);
  Sampler b(cfg);
  EXPECT_EQ(a.sample(logits), b.sample(logits));
}

TEST(SamplerTest, InvalidTemperatureThrows) {
  SamplerConfig cfg;
  cfg.temperature = 0.f;
  EXPECT_THROW({ Sampler s(cfg); }, invalid_argument);
}

}  // namespace serving
