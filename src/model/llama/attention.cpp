#include "model/llama/attention.hpp"

#include "cache/kv_cache.hpp"
#include "ops/ops.hpp"
#include "ops/rope.hpp"

#include <cmath>
#include <stdexcept>

using namespace std;

namespace serving {

namespace {

// reshape_proj: Tensor — views linear output as [batch, seq, heads, head_dim].
Tensor reshape_proj(const Tensor& flat, int64_t batch, int64_t seq, int64_t heads,
                    int64_t head_dim) {
  if (flat.shape() != vector<int64_t>{batch * seq, heads * head_dim}) {
    throw invalid_argument("proj flat shape");
  }
  return flat.view({batch, seq, heads, head_dim});
}

// attn_scores_row: float — dot product over head_dim for one query/key head vector.
float attn_scores_row(const Tensor& q, const Tensor& k, int64_t b, int64_t hq, int64_t tq,
                      int64_t hk, int64_t tk) {
  int64_t head_dim = q.shape()[3];
  float sum = 0.f;
  for (int64_t d = 0; d < head_dim; ++d) {
    sum += q.at({b, tq, hq, d}) * k.at({b, tk, hk, d});
  }
  return sum;
}

// write_attn_output: void — causal softmax attention from q,k,v into attn_out slice.
void write_attn_output(const Tensor& q, const Tensor& k, const Tensor& v, int64_t batch,
                       int64_t seq, int64_t n_heads, int64_t n_kv_heads, int64_t head_dim,
                       float scale, Tensor& attn_out) {
  int64_t n_rep = n_heads / n_kv_heads;
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t tq = 0; tq < seq; ++tq) {
      for (int64_t hq = 0; hq < n_heads; ++hq) {
        int64_t hk = hq / n_rep;
        float max_score = -1e30f;
        vector<float> scores(static_cast<size_t>(tq + 1));
        for (int64_t tk = 0; tk <= tq; ++tk) {
          float s = attn_scores_row(q, k, b, hq, tq, hk, tk) * scale;
          scores[static_cast<size_t>(tk)] = s;
          max_score = max(max_score, s);
        }
        float sum = 0.f;
        for (int64_t tk = 0; tk <= tq; ++tk) {
          float e = expf(scores[static_cast<size_t>(tk)] - max_score);
          scores[static_cast<size_t>(tk)] = e;
          sum += e;
        }
        for (int64_t d = 0; d < head_dim; ++d) {
          float acc = 0.f;
          for (int64_t tk = 0; tk <= tq; ++tk) {
            float w = scores[static_cast<size_t>(tk)] / sum;
            acc += w * v.at({b, tk, hk, d});
          }
          attn_out.at({b, tq, hq, d}) = acc;
        }
      }
    }
  }
}

// merge_attn_and_linear: void — merges heads and applies output projection into out.
void merge_attn_and_linear(const Tensor& attn_out, const AttentionWeights& weights, int64_t batch,
                           int64_t seq, int64_t dim, int64_t n_heads, int64_t head_dim,
                           Tensor& out) {
  Tensor merged({batch * seq, dim});
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t s = 0; s < seq; ++s) {
      int64_t row = b * seq + s;
      for (int64_t h = 0; h < n_heads; ++h) {
        for (int64_t d = 0; d < head_dim; ++d) {
          merged.at({row, h * head_dim + d}) = attn_out.at({b, s, h, d});
        }
      }
    }
  }
  Tensor out2d({batch * seq, dim});
  linear(merged, weights.wo, out2d);
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t s = 0; s < seq; ++s) {
      for (int64_t d = 0; d < dim; ++d) {
        out.at({b, s, d}) = out2d.at({b * seq + s, d});
      }
    }
  }
}

}  // namespace

// gqa_attention: void — causal GQA attention; x[B,S,D] -> out[B,S,D] (full sequence, no KV cache).
void gqa_attention(const Tensor& x, const AttentionWeights& weights, const BlockConfig& config,
                   const vector<int64_t>& positions, Tensor& out) {
  const auto& xs = x.shape();
  if (xs.size() != 3) {
    throw invalid_argument("attention x expects [batch, seq, dim]");
  }
  if (out.shape() != xs) {
    throw invalid_argument("attention out shape");
  }
  int64_t batch = xs[0];
  int64_t seq = xs[1];
  int64_t dim = xs[2];
  if (dim != config.dim) {
    throw invalid_argument("attention dim mismatch");
  }
  if (config.n_heads % config.n_kv_heads != 0) {
    throw invalid_argument("n_heads must be divisible by n_kv_heads");
  }
  int64_t n_rep = config.n_heads / config.n_kv_heads;
  float safety_cap = 1.f / sqrtf(static_cast<float>(config.head_dim));

  Tensor x2d = x.view({batch * seq, dim});
  Tensor q_flat({batch * seq, dim});
  Tensor k_flat({batch * seq, config.n_kv_heads * config.head_dim});
  Tensor v_flat({batch * seq, config.n_kv_heads * config.head_dim});
  linear(x2d, weights.wq, q_flat);
  linear(x2d, weights.wk, k_flat);
  linear(x2d, weights.wv, v_flat);

  Tensor q = reshape_proj(q_flat, batch, seq, config.n_heads, config.head_dim);
  Tensor k = reshape_proj(k_flat, batch, seq, config.n_kv_heads, config.head_dim);
  Tensor v = reshape_proj(v_flat, batch, seq, config.n_kv_heads, config.head_dim);

  apply_rope(q, k, positions, config.rope_theta);

  Tensor attn_out({batch, seq, config.n_heads, config.head_dim});

  write_attn_output(q, k, v, batch, seq, config.n_heads, config.n_kv_heads, config.head_dim,
                    safety_cap, attn_out);
  merge_attn_and_linear(attn_out, weights, batch, seq, dim, config.n_heads, config.head_dim, out);

  // for (int64_t b = 0; b < batch; ++b) {
  //   for (int64_t tq = 0; tq < seq; ++tq) {
  //     for (int64_t hq = 0; hq < config.n_heads; ++hq) {
  //       int64_t hk = hq / n_rep;
  //       float max_score = -1e30f;
  //       vector<float> scores(static_cast<size_t>(tq + 1));
  //       for (int64_t tk = 0; tk <= tq; ++tk) {
  //         float s = attn_scores_row(q, k, b, hq, tq, hk, tk) * safety_cap;
  //         scores[static_cast<size_t>(tk)] = s;
  //         max_score = max(max_score, s);
  //       }
  //       float sum = 0.f;
  //       for (int64_t tk = 0; tk <= tq; ++tk) {
  //         float e = expf(scores[static_cast<size_t>(tk)] - max_score);
  //         scores[static_cast<size_t>(tk)] = e;
  //         sum += e;
  //       }
  //       for (int64_t d = 0; d < config.head_dim; ++d) {
  //         float acc = 0.f;
  //         for (int64_t tk = 0; tk <= tq; ++tk) {
  //           float w = scores[static_cast<size_t>(tk)] / sum;
  //           acc += w * v.at({b, tk, hk, d});
  //         }
  //         attn_out.at({b, tq, hq, d}) = acc;
  //       }
  //     }
  //   }
  // }

  // Tensor merged({batch * seq, dim});
  // for (int64_t b = 0; b < batch; ++b) {
  //   for (int64_t s = 0; s < seq; ++s) {
  //     int64_t row = b * seq + s;
  //     for (int64_t h = 0; h < config.n_heads; ++h) {
  //       for (int64_t d = 0; d < config.head_dim; ++d) {
  //         merged.at({row, h * config.head_dim + d}) = attn_out.at({b, s, h, d});
  //       }
  //     }
  //   }
  // }
  // Tensor out2d({batch * seq, dim});
  // linear(merged, weights.wo, out2d);
  // for (int64_t b = 0; b < batch; ++b) {
  //   for (int64_t s = 0; s < seq; ++s) {
  //     for (int64_t d = 0; d < dim; ++d) {
  //       out.at({b, s, d}) = out2d.at({b * seq + s, d});
  //     }
  //   }
  // }
}

// gqa_attention_prefill: void — same as gqa_attention; appends post-RoPE K/V into cache.
void gqa_attention_prefill(const Tensor& x, const AttentionWeights& weights,
                             const BlockConfig& config, const vector<int64_t>& positions,
                             KVCache& cache, Tensor& out) {
  const auto& xs = x.shape();
  if (xs.size() != 3) {
    throw invalid_argument("attention x expects [batch, seq, dim]");
  }
  if (out.shape() != xs) {
    throw invalid_argument("attention out shape");
  }
  int64_t batch = xs[0];
  int64_t seq = xs[1];
  int64_t dim = xs[2];
  if (dim != config.dim) {
    throw invalid_argument("attention dim mismatch");
  }
  if (config.n_heads % config.n_kv_heads != 0) {
    throw invalid_argument("n_heads must be divisible by n_kv_heads");
  }
  if (batch != cache.batch()) {
    throw invalid_argument("KVCache batch mismatch");
  }
  float safety_cap = 1.f / sqrtf(static_cast<float>(config.head_dim));

  Tensor x2d = x.view({batch * seq, dim});
  Tensor q_flat({batch * seq, dim});
  Tensor k_flat({batch * seq, config.n_kv_heads * config.head_dim});
  Tensor v_flat({batch * seq, config.n_kv_heads * config.head_dim});
  linear(x2d, weights.wq, q_flat);
  linear(x2d, weights.wk, k_flat);
  linear(x2d, weights.wv, v_flat);

  Tensor q = reshape_proj(q_flat, batch, seq, config.n_heads, config.head_dim);
  Tensor k = reshape_proj(k_flat, batch, seq, config.n_kv_heads, config.head_dim);
  Tensor v = reshape_proj(v_flat, batch, seq, config.n_kv_heads, config.head_dim);

  apply_rope(q, k, positions, config.rope_theta);
  cache.append(k, v);

  Tensor attn_out({batch, seq, config.n_heads, config.head_dim});
  write_attn_output(q, k, v, batch, seq, config.n_heads, config.n_kv_heads, config.head_dim,
                    safety_cap, attn_out);
  merge_attn_and_linear(attn_out, weights, batch, seq, dim, config.n_heads, config.head_dim, out);
}

// gqa_attention_decode: void — one token x[B,1,D]; uses cache and absolute position.
void gqa_attention_decode(const Tensor& x, const AttentionWeights& weights,
                          const BlockConfig& config, int64_t position, KVCache& cache,
                          Tensor& out) {
  const auto& xs = x.shape();
  if (xs.size() != 3 || xs[1] != 1) {
    throw invalid_argument("decode x expects [batch, 1, dim]");
  }
  if (out.shape() != xs) {
    throw invalid_argument("attention out shape");
  }
  int64_t batch = xs[0];
  int64_t dim = xs[2];
  if (dim != config.dim) {
    throw invalid_argument("attention dim mismatch");
  }
  if (config.n_heads % config.n_kv_heads != 0) {
    throw invalid_argument("n_heads must be divisible by n_kv_heads");
  }
  if (batch != cache.batch()) {
    throw invalid_argument("KVCache batch mismatch");
  }
  int64_t n_rep = config.n_heads / config.n_kv_heads;
  float scale = 1.f / sqrtf(static_cast<float>(config.head_dim));

  Tensor x2d = x.view({batch, dim});
  Tensor q_flat({batch, dim});
  Tensor k_flat({batch, config.n_kv_heads * config.head_dim});
  Tensor v_flat({batch, config.n_kv_heads * config.head_dim});
  linear(x2d, weights.wq, q_flat);
  linear(x2d, weights.wk, k_flat);
  linear(x2d, weights.wv, v_flat);

  Tensor q = reshape_proj(q_flat, batch, 1, config.n_heads, config.head_dim);
  Tensor k = reshape_proj(k_flat, batch, 1, config.n_kv_heads, config.head_dim);
  Tensor v = reshape_proj(v_flat, batch, 1, config.n_kv_heads, config.head_dim);

  vector<int64_t> pos = {position};
  apply_rope(q, k, pos, config.rope_theta);
  cache.append(k, v);

  const Tensor& k_cache = cache.keys();
  const Tensor& v_cache = cache.values();
  int64_t cache_len = cache.seq_len();

  Tensor attn_out({batch, 1, config.n_heads, config.head_dim});
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t hq = 0; hq < config.n_heads; ++hq) {
      int64_t hk = hq / n_rep;
      float max_score = -1e30f;
      vector<float> scores(static_cast<size_t>(cache_len));
      for (int64_t tk = 0; tk < cache_len; ++tk) {
        float s = attn_scores_row(q, k_cache, b, hq, 0, hk, tk) * scale;
        scores[static_cast<size_t>(tk)] = s;
        max_score = max(max_score, s);
      }
      float sum = 0.f;
      for (int64_t tk = 0; tk < cache_len; ++tk) {
        float e = expf(scores[static_cast<size_t>(tk)] - max_score);
        scores[static_cast<size_t>(tk)] = e;
        sum += e;
      }
      for (int64_t d = 0; d < config.head_dim; ++d) {
        float acc = 0.f;
        for (int64_t tk = 0; tk < cache_len; ++tk) {
          float w = scores[static_cast<size_t>(tk)] / sum;
          acc += w * v_cache.at({b, tk, hk, d});
        }
        attn_out.at({b, 0, hq, d}) = acc;
      }
    }
  }
  merge_attn_and_linear(attn_out, weights, batch, 1, dim, config.n_heads, config.head_dim, out);
}

}  // namespace serving
