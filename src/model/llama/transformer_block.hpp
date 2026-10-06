#pragma once

#include "core/tensor.hpp"
#include "model/llama/attention.hpp"
#include "model/llama/block_config.hpp"

#include <vector>

namespace serving {

struct TransformerBlockWeights {
  Tensor attn_norm;
  AttentionWeights attn;
  Tensor ffn_norm;
  Tensor w_gate;
  Tensor w_up;
  Tensor w_down;
};

class TransformerBlock {
 public:
  // TransformerBlock: — holds config and weights for one Llama-style layer.
  TransformerBlock(BlockConfig config, TransformerBlockWeights weights);

  // forward: void — full-sequence block forward (Phase 2a, no KV cache).
  void forward(const Tensor& x, const std::vector<int64_t>& positions, Tensor& out) const;

  // config: const BlockConfig& — returns block hyperparameters.
  const BlockConfig& config() const { return config_; }

 private:
  BlockConfig config_;
  TransformerBlockWeights weights_;
};

}  // namespace serving
