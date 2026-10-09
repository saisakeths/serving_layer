#include "cache/kv_cache.hpp"
#include "model/llama/block_config.hpp"
#include "model/llama/transformer_block.hpp"

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

TransformerBlockWeights make_weights(const BlockConfig& cfg) {
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
  fill_pattern(w.attn.wq, 0.01f);
  fill_pattern(w.attn.wk, 0.01f);
  fill_pattern(w.attn.wv, 0.01f);
  fill_pattern(w.attn.wo, 0.01f);
  fill_pattern(w.w_gate, 0.01f);
  fill_pattern(w.w_up, 0.01f);
  fill_pattern(w.w_down, 0.01f);
  return w;
}

vector<int64_t> positions_for(int64_t seq) {
  vector<int64_t> pos(static_cast<size_t>(seq));
  for (int64_t i = 0; i < seq; ++i) {
    pos[static_cast<size_t>(i)] = i;
  }
  return pos;
}

TEST(KVCacheTest, PrefillWithCacheMatchesForward) {
  BlockConfig cfg = tiny_config();
  TransformerBlock block(cfg, make_weights(cfg));
  int64_t batch = 1;
  int64_t seq = 4;
  Tensor x({batch, seq, cfg.dim});
  fill_pattern(x, 0.05f);
  auto pos = positions_for(seq);
  KVCache cache(batch, 16, cfg.n_kv_heads, cfg.head_dim);
  Tensor out_prefill({batch, seq, cfg.dim});
  Tensor out_forward({batch, seq, cfg.dim});
  block.forward_prefill(x, pos, cache, out_prefill);
  block.forward(x, pos, out_forward);
  for (size_t i = 0; i < out_prefill.numel(); ++i) {
    EXPECT_NEAR(out_prefill.data()[i], out_forward.data()[i], 1e-4f);
  }
  EXPECT_EQ(cache.seq_len(), seq);
}

TEST(KVCacheTest, DecodeMatchesJointPrefill) {
  BlockConfig cfg = tiny_config();
  TransformerBlock block(cfg, make_weights(cfg));
  int64_t batch = 1;
  int64_t P = 2;
  int64_t total = P + 2;
  Tensor x_joint({batch, total, cfg.dim});
  fill_pattern(x_joint, 0.04f);
  auto pos_joint = positions_for(total);
  Tensor out_joint({batch, total, cfg.dim});
  block.forward(x_joint, pos_joint, out_joint);

  Tensor x_prompt({batch, P, cfg.dim});
  for (int64_t s = 0; s < P; ++s) {
    for (int64_t d = 0; d < cfg.dim; ++d) {
      x_prompt.at({0, s, d}) = x_joint.at({0, s, d});
    }
  }
  Tensor tok1({batch, 1, cfg.dim});
  Tensor tok2({batch, 1, cfg.dim});
  for (int64_t d = 0; d < cfg.dim; ++d) {
    tok1.at({0, 0, d}) = x_joint.at({0, P, d});
    tok2.at({0, 0, d}) = x_joint.at({0, P + 1, d});
  }

  KVCache cache(batch, 16, cfg.n_kv_heads, cfg.head_dim);
  Tensor out_prompt({batch, P, cfg.dim});
  block.forward_prefill(x_prompt, positions_for(P), cache, out_prompt);
  Tensor out1({batch, 1, cfg.dim});
  Tensor out2({batch, 1, cfg.dim});
  block.decode_step(tok1, P, cache, out1);
  block.decode_step(tok2, P + 1, cache, out2);

  for (int64_t d = 0; d < cfg.dim; ++d) {
    EXPECT_NEAR(out1.at({0, 0, d}), out_joint.at({0, P, d}), 1e-4f);
    EXPECT_NEAR(out2.at({0, 0, d}), out_joint.at({0, P + 1, d}), 1e-4f);
  }
}

TEST(KVCacheTest, CacheLength) {
  BlockConfig cfg = tiny_config();
  TransformerBlock block(cfg, make_weights(cfg));
  int64_t batch = 1;
  int64_t S = 3;
  int64_t T = 2;
  Tensor x({batch, S, cfg.dim});
  fill_pattern(x, 0.03f);
  KVCache cache(batch, 16, cfg.n_kv_heads, cfg.head_dim);
  Tensor out({batch, S, cfg.dim});
  block.forward_prefill(x, positions_for(S), cache, out);
  EXPECT_EQ(cache.seq_len(), S);
  Tensor tok({batch, 1, cfg.dim});
  fill_pattern(tok, 0.02f);
  Tensor step_out({batch, 1, cfg.dim});
  for (int64_t t = 0; t < T; ++t) {
    block.decode_step(tok, S + t, cache, step_out);
  }
  EXPECT_EQ(cache.seq_len(), S + T);
}

TEST(KVCacheTest, DecodeDeterministic) {
  BlockConfig cfg = tiny_config();
  TransformerBlock block(cfg, make_weights(cfg));
  Tensor x({1, 2, cfg.dim});
  fill_pattern(x, 0.035f);
  Tensor tok({1, 1, cfg.dim});
  fill_pattern(tok, 0.025f);
  KVCache cache_a(1, 8, cfg.n_kv_heads, cfg.head_dim);
  KVCache cache_b(1, 8, cfg.n_kv_heads, cfg.head_dim);
  Tensor oa({1, 1, cfg.dim});
  Tensor ob({1, 1, cfg.dim});
  Tensor pref_a({1, 2, cfg.dim});
  Tensor pref_b({1, 2, cfg.dim});
  block.forward_prefill(x, positions_for(2), cache_a, pref_a);
  block.forward_prefill(x, positions_for(2), cache_b, pref_b);
  block.decode_step(tok, 2, cache_a, oa);
  block.decode_step(tok, 2, cache_b, ob);
  for (size_t i = 0; i < oa.numel(); ++i) {
    EXPECT_NEAR(oa.data()[i], ob.data()[i], 1e-5f);
  }
}

}  // namespace
}  // namespace serving
