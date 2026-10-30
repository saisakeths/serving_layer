#include "model/llama/llama_cache.hpp"
#include "model/llama/llama_model.hpp"
#include "sampler/generation_loop.hpp"
#include "sampler/sampler.hpp"

#include <gtest/gtest.h>

using namespace std;

namespace serving {

TEST(GenerationTest, GreedyTokenIdsGolden) {
  LlamaConfig cfg = LlamaConfig::tiny();
  LlamaWeights w = make_random_llama_weights(cfg, 0.01f);
  LlamaModel model(cfg, move(w));
  vector<int32_t> prompt = {1, 2, 3};
  const int max_new = 4;
  LlamaCache cache(cfg, 1, static_cast<int64_t>(prompt.size() + max_new));
  SamplerConfig scfg;
  scfg.greedy = true;
  Sampler sampler(scfg);
  GenerationConfig gcfg;
  gcfg.max_tokens = max_new;
  gcfg.eos_token_id = -1;
  vector<int32_t> out = generate(model, cache, prompt, sampler, gcfg);
  EXPECT_EQ(out.size(), static_cast<size_t>(max_new));
  vector<int32_t> expected = {0, 0, 0, 0};
  EXPECT_EQ(out, expected);
  LlamaCache cache2(cfg, 1, static_cast<int64_t>(prompt.size() + max_new));
  vector<int32_t> again = generate(model, cache2, prompt, sampler, gcfg);
  EXPECT_EQ(again, expected);
}

TEST(GenerationTest, StopsOnEos) {
  LlamaConfig cfg = LlamaConfig::tiny();
  LlamaWeights w = make_random_llama_weights(cfg, 0.01f);
  LlamaModel model(cfg, move(w));
  vector<int32_t> prompt = {1, 2, 3};
  LlamaCache cache(cfg, 1, 128);
  SamplerConfig scfg;
  scfg.greedy = true;
  Sampler sampler(scfg);
  GenerationConfig gcfg;
  gcfg.max_tokens = 100;
  gcfg.eos_token_id = 0;
  vector<int32_t> out = generate(model, cache, prompt, sampler, gcfg);
  EXPECT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0], 0);
}
}  // namespace serving
