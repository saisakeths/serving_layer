#include "model/llama/transformer_block.hpp"

#include <gtest/gtest.h>

using namespace std;

namespace serving {
namespace {

TransformerBlockWeights make_tiny_weights(const BlockConfig& cfg) {
  TransformerBlockWeights w;
  w.attn_norm = Tensor({cfg.dim});
  w.ffn_norm = Tensor({cfg.dim});
  for (int64_t i = 0; i < cfg.dim; ++i) {
    w.attn_norm.at({i}) = 1.f;
    w.ffn_norm.at({i}) = 1.f;
  }
  w.attn.wq = Tensor({cfg.dim, cfg.dim});
  w.attn.wk = Tensor({cfg.n_kv_heads * cfg.head_dim, cfg.dim});
  w.attn.wv = Tensor({cfg.n_kv_heads * cfg.head_dim, cfg.dim});
  w.attn.wo = Tensor({cfg.dim, cfg.dim});
  w.w_gate = Tensor({cfg.ffn_dim, cfg.dim});
  w.w_up = Tensor({cfg.ffn_dim, cfg.dim});
  w.w_down = Tensor({cfg.dim, cfg.ffn_dim});
  for (size_t i = 0; i < w.attn.wq.numel(); ++i) {
    w.attn.wq.data()[i] = 0.01f * static_cast<float>(static_cast<int>(i % 7) - 3);
  }
  for (size_t i = 0; i < w.attn.wk.numel(); ++i) {
    w.attn.wk.data()[i] = 0.01f;
  }
  for (size_t i = 0; i < w.attn.wv.numel(); ++i) {
    w.attn.wv.data()[i] = 0.01f;
  }
  for (size_t i = 0; i < w.attn.wo.numel(); ++i) {
    w.attn.wo.data()[i] = 0.01f;
  }
  for (size_t i = 0; i < w.w_gate.numel(); ++i) {
    w.w_gate.data()[i] = 0.005f;
  }
  for (size_t i = 0; i < w.w_up.numel(); ++i) {
    w.w_up.data()[i] = 0.005f;
  }
  for (size_t i = 0; i < w.w_down.numel(); ++i) {
    w.w_down.data()[i] = 0.005f;
  }
  return w;
}

BlockConfig tiny_config() {
  BlockConfig c;
  c.dim = 8;
  c.n_heads = 2;
  c.n_kv_heads = 1;
  c.head_dim = 4;
  c.ffn_dim = 16;
  return c;
}

TEST(OneBlockForwardTest, ShapeAndFinite) {
  BlockConfig cfg = tiny_config();
  TransformerBlock block(cfg, make_tiny_weights(cfg));
  Tensor x({1, 4, cfg.dim});
  for (size_t i = 0; i < x.numel(); ++i) {
    x.data()[i] = 0.1f * static_cast<float>(i);
  }
  vector<int64_t> pos = {0, 1, 2, 3};
  Tensor out({1, 4, cfg.dim});
  block.forward(x, pos, out);
  EXPECT_EQ(out.shape(), x.shape());
  EXPECT_FALSE(std::isnan(out.at({0, 3, 0})));
}

TEST(OneBlockForwardTest, Deterministic) {
  BlockConfig cfg = tiny_config();
  TransformerBlock block(cfg, make_tiny_weights(cfg));
  Tensor x({1, 3, cfg.dim});
  for (size_t i = 0; i < x.numel(); ++i) {
    x.data()[i] = 0.02f * static_cast<float>(i);
  }
  vector<int64_t> pos = {0, 1, 2};
  Tensor a({1, 3, cfg.dim});
  Tensor b({1, 3, cfg.dim});
  block.forward(x, pos, a);
  block.forward(x, pos, b);
  for (size_t i = 0; i < a.numel(); ++i) {
    EXPECT_NEAR(a.data()[i], b.data()[i], 1e-5f);
  }
}

}  // namespace
}  // namespace serving
