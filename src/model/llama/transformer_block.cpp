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

}  // namespace serving
