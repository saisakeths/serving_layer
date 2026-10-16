#include "model/llama/llama_cache.hpp"
#include "model/llama/llama_model.hpp"

#include <gtest/gtest.h>

#include <cmath>

using namespace std;

namespace serving {

TEST(LlamaForwardTest, ForwardLogitsShape) {
  LlamaConfig cfg = LlamaConfig::tiny();
  LlamaWeights w = make_random_llama_weights(cfg, 0.01f);
  LlamaModel model(cfg, move(w));
  vector<int32_t> ids = {1, 2, 3, 4};
  vector<int64_t> pos = {0, 1, 2, 3};
  LlamaCache cache(cfg, 1, 4);
  Tensor logits({1, 4, cfg.vocab_size});
  model.forward(ids, pos, cache, logits);
  EXPECT_EQ(logits.shape(), vector<int64_t>({1, 4, cfg.vocab_size}));
  EXPECT_FALSE(isnan(logits.at({0, 3, 0})));
}

TEST(LlamaForwardTest, ForwardDeterministic) {
  LlamaConfig cfg = LlamaConfig::tiny();
  LlamaWeights w = make_random_llama_weights(cfg, 0.01f);
  LlamaModel model(cfg, move(w));
  vector<int32_t> ids = {0, 5, 10};
  vector<int64_t> pos = {0, 1, 2};
  Tensor a({1, 3, cfg.vocab_size});
  Tensor b({1, 3, cfg.vocab_size});
  LlamaCache cache(cfg, 1, 3);
  model.forward(ids, pos, cache, a);
  model.forward(ids, pos, cache, b);
  for (size_t i = 0; i < a.numel(); ++i) {
    EXPECT_NEAR(a.data()[i], b.data()[i], 1e-5f);
  }
}

}  // namespace serving
