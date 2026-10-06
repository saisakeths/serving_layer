#include "model/llama/attention.hpp"
#include "model/llama/block_config.hpp"
#include <cmath>
#include <gtest/gtest.h>

using namespace std;

namespace serving {
namespace {

BlockConfig tiny_config() {
  BlockConfig c;
  c.dim = 8;
  c.n_heads = 2;
  c.n_kv_heads = 1;
  c.head_dim = 4;
  c.ffn_dim = 16;
  c.rope_theta = 10000.f;
  return c;
}

void fill_pattern(Tensor& t, float scale) {
  for (size_t i = 0; i < t.numel(); ++i) {
    t.data()[i] = scale * static_cast<float>(static_cast<int>(i % 11) - 5);
  }
}

TEST(AttentionTest, OutputShape) {
  BlockConfig cfg = tiny_config();
  int64_t batch = 1;
  int64_t seq = 4;
  Tensor x({batch, seq, cfg.dim});
  fill_pattern(x, 0.05f);
  AttentionWeights w;
  w.wq = Tensor({cfg.dim, cfg.dim});
  w.wk = Tensor({cfg.n_kv_heads * cfg.head_dim, cfg.dim});
  w.wv = Tensor({cfg.n_kv_heads * cfg.head_dim, cfg.dim});
  w.wo = Tensor({cfg.dim, cfg.dim});
  fill_pattern(w.wq, 0.01f);
  fill_pattern(w.wk, 0.01f);
  fill_pattern(w.wv, 0.01f);
  fill_pattern(w.wo, 0.01f);
  vector<int64_t> pos = {0, 1, 2, 3};
  Tensor out({batch, seq, cfg.dim});
  gqa_attention(x, w, cfg, pos, out);
  EXPECT_EQ(out.shape(), x.shape());
  EXPECT_FALSE(std::isnan(out.at({0, 0, 0})));
}

TEST(AttentionTest, Deterministic) {
  BlockConfig cfg = tiny_config();
  Tensor x({1, 4, cfg.dim});
  fill_pattern(x, 0.03f);
  AttentionWeights w;
  w.wq = Tensor({cfg.dim, cfg.dim});
  w.wk = Tensor({cfg.n_kv_heads * cfg.head_dim, cfg.dim});
  w.wv = Tensor({cfg.n_kv_heads * cfg.head_dim, cfg.dim});
  w.wo = Tensor({cfg.dim, cfg.dim});
  fill_pattern(w.wq, 0.02f);
  fill_pattern(w.wk, 0.02f);
  fill_pattern(w.wv, 0.02f);
  fill_pattern(w.wo, 0.02f);
  vector<int64_t> pos = {0, 1, 2, 3};
  Tensor a({1, 4, cfg.dim});
  Tensor b({1, 4, cfg.dim});
  gqa_attention(x, w, cfg, pos, a);
  gqa_attention(x, w, cfg, pos, b);
  for (size_t i = 0; i < a.numel(); ++i) {
    EXPECT_NEAR(a.data()[i], b.data()[i], 1e-5f);
  }
}

TEST(AttentionTest, CausalPrefixStableAcrossSeqLen) {
  BlockConfig cfg = tiny_config();
  Tensor x4({1, 4, cfg.dim});
  fill_pattern(x4, 0.04f);
  Tensor x2({1, 2, cfg.dim});
  for (int64_t s = 0; s < 2; ++s) {
    for (int64_t d = 0; d < cfg.dim; ++d) {
      x2.at({0, s, d}) = x4.at({0, s, d});
    }
  }
  AttentionWeights w;
  w.wq = Tensor({cfg.dim, cfg.dim});
  w.wk = Tensor({cfg.n_kv_heads * cfg.head_dim, cfg.dim});
  w.wv = Tensor({cfg.n_kv_heads * cfg.head_dim, cfg.dim});
  w.wo = Tensor({cfg.dim, cfg.dim});
  fill_pattern(w.wq, 0.015f);
  fill_pattern(w.wk, 0.015f);
  fill_pattern(w.wv, 0.015f);
  fill_pattern(w.wo, 0.015f);
  vector<int64_t> pos4 = {0, 1, 2, 3};
  vector<int64_t> pos2 = {0, 1};
  Tensor out4({1, 4, cfg.dim});
  Tensor out2({1, 2, cfg.dim});
  gqa_attention(x4, w, cfg, pos4, out4);
  gqa_attention(x2, w, cfg, pos2, out2);
  for (int64_t d = 0; d < cfg.dim; ++d) {
    EXPECT_NEAR(out4.at({0, 1, d}), out2.at({0, 1, d}), 1e-4f);
  }
}

}  // namespace
}  // namespace serving
