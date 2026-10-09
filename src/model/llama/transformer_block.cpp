#include "model/llama/transformer_block.hpp"

#include "ops/ops.hpp"

#include <stdexcept>

using namespace std;

namespace serving {

// TransformerBlock: — holds config and weights for one Llama-style layer.
TransformerBlock::TransformerBlock(BlockConfig config, TransformerBlockWeights weights)
    : config_(config), weights_(move(weights)) {}

// forward: void — full-sequence block forward (Phase 2a, no KV cache).
void TransformerBlock::forward(const Tensor& x, const vector<int64_t>& positions, Tensor& out) const {
  const auto& xs = x.shape();
  if (xs.size() != 3 || xs[2] != config_.dim) {
    throw invalid_argument("block x shape");
  }
  if (out.shape() != xs) {
    throw invalid_argument("block out shape");
  }
  int64_t batch = xs[0];
  int64_t seq = xs[1];
  int64_t dim = xs[2];

  Tensor x2d = x.view({batch * seq, dim});
  Tensor normed2d({batch * seq, dim});
  rmsnorm(x2d, weights_.attn_norm, config_.rms_eps, normed2d);
  Tensor normed = normed2d.view({batch, seq, dim});

  Tensor attn_out({batch, seq, dim});
  gqa_attention(normed, weights_.attn, config_, positions, attn_out);

  Tensor residual1({batch, seq, dim});
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t s = 0; s < seq; ++s) {
      for (int64_t d = 0; d < dim; ++d) {
        residual1.at({b, s, d}) = x.at({b, s, d}) + attn_out.at({b, s, d});
      }
    }
  }

  Tensor res2d = residual1.view({batch * seq, dim});
  Tensor ffn_normed2d({batch * seq, dim});
  rmsnorm(res2d, weights_.ffn_norm, config_.rms_eps, ffn_normed2d);
  Tensor gate({batch * seq, config_.ffn_dim});
  Tensor up({batch * seq, config_.ffn_dim});
  linear(ffn_normed2d, weights_.w_gate, gate);
  linear(ffn_normed2d, weights_.w_up, up);
  silu_inplace(gate);
  mul_inplace(gate, up);

  Tensor ffn_down({batch * seq, dim});
  linear(gate, weights_.w_down, ffn_down);

  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t s = 0; s < seq; ++s) {
      for (int64_t d = 0; d < dim; ++d) {
        out.at({b, s, d}) = residual1.at({b, s, d}) + ffn_down.at({b * seq + s, d});
      }
    }
  }
}

// forward_prefill: void — like forward; resets cache and fills it with post-RoPE K/V.
void TransformerBlock::forward_prefill(const Tensor& x, const vector<int64_t>& positions,
                                       KVCache& cache, Tensor& out) const {
  cache.reset();
  const auto& xs = x.shape();
  if (xs.size() != 3 || xs[2] != config_.dim) {
    throw invalid_argument("block x shape");
  }
  if (out.shape() != xs) {
    throw invalid_argument("block out shape");
  }
  int64_t batch = xs[0];
  int64_t seq = xs[1];
  int64_t dim = xs[2];

  Tensor x2d = x.view({batch * seq, dim});
  Tensor normed2d({batch * seq, dim});
  rmsnorm(x2d, weights_.attn_norm, config_.rms_eps, normed2d);
  Tensor normed = normed2d.view({batch, seq, dim});

  Tensor attn_out({batch, seq, dim});
  gqa_attention_prefill(normed, weights_.attn, config_, positions, cache, attn_out);

  Tensor residual1({batch, seq, dim});
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t s = 0; s < seq; ++s) {
      for (int64_t d = 0; d < dim; ++d) {
        residual1.at({b, s, d}) = x.at({b, s, d}) + attn_out.at({b, s, d});
      }
    }
  }

  Tensor res2d = residual1.view({batch * seq, dim});
  Tensor ffn_normed2d({batch * seq, dim});
  rmsnorm(res2d, weights_.ffn_norm, config_.rms_eps, ffn_normed2d);
  Tensor gate({batch * seq, config_.ffn_dim});
  Tensor up({batch * seq, config_.ffn_dim});
  linear(ffn_normed2d, weights_.w_gate, gate);
  linear(ffn_normed2d, weights_.w_up, up);
  silu_inplace(gate);
  mul_inplace(gate, up);

  Tensor ffn_down({batch * seq, dim});
  linear(gate, weights_.w_down, ffn_down);

  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t s = 0; s < seq; ++s) {
      for (int64_t d = 0; d < dim; ++d) {
        out.at({b, s, d}) = residual1.at({b, s, d}) + ffn_down.at({b * seq + s, d});
      }
    }
  }
}

// decode_step: void — one token x[B,1,D] at absolute position.
void TransformerBlock::decode_step(const Tensor& x, int64_t position, KVCache& cache,
                                   Tensor& out) const {
  const auto& xs = x.shape();
  if (xs.size() != 3 || xs[1] != 1 || xs[2] != config_.dim) {
    throw invalid_argument("decode_step x expects [batch, 1, dim]");
  }
  if (out.shape() != xs) {
    throw invalid_argument("block out shape");
  }
  int64_t batch = xs[0];
  int64_t seq = 1;
  int64_t dim = xs[2];

  Tensor x2d = x.view({batch, dim});
  Tensor normed2d({batch, dim});
  rmsnorm(x2d, weights_.attn_norm, config_.rms_eps, normed2d);
  Tensor normed = normed2d.view({batch, seq, dim});

  Tensor attn_out({batch, seq, dim});
  gqa_attention_decode(normed, weights_.attn, config_, position, cache, attn_out);

  Tensor residual1({batch, seq, dim});
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t d = 0; d < dim; ++d) {
      residual1.at({b, 0, d}) = x.at({b, 0, d}) + attn_out.at({b, 0, d});
    }
  }

  Tensor res2d = residual1.view({batch, dim});
  Tensor ffn_normed2d({batch, dim});
  rmsnorm(res2d, weights_.ffn_norm, config_.rms_eps, ffn_normed2d);
  Tensor gate({batch, config_.ffn_dim});
  Tensor up({batch, config_.ffn_dim});
  linear(ffn_normed2d, weights_.w_gate, gate);
  linear(ffn_normed2d, weights_.w_up, up);
  silu_inplace(gate);
  mul_inplace(gate, up);

  Tensor ffn_down({batch, dim});
  linear(gate, weights_.w_down, ffn_down);

  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t d = 0; d < dim; ++d) {
      out.at({b, 0, d}) = residual1.at({b, 0, d}) + ffn_down.at({b, d});
    }
  }
}

}  // namespace serving
