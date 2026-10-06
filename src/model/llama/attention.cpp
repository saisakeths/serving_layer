#include "model/llama/attention.hpp"

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
  float scale = 1.f / sqrtf(static_cast<float>(config.head_dim));

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
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t tq = 0; tq < seq; ++tq) {
      for (int64_t hq = 0; hq < config.n_heads; ++hq) {
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
        for (int64_t d = 0; d < config.head_dim; ++d) {
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

  Tensor merged({batch * seq, dim});
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t s = 0; s < seq; ++s) {
      int64_t row = b * seq + s;
      for (int64_t h = 0; h < config.n_heads; ++h) {
        for (int64_t d = 0; d < config.head_dim; ++d) {
          merged.at({row, h * config.head_dim + d}) = attn_out.at({b, s, h, d});
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

}  // namespace serving
