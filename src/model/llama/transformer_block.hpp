#pragma once

#include "cache/kv_cache.hpp"
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

  // forward_prefill: void — like forward; resets cache and fills it with post-RoPE K/V.
  void forward_prefill(const Tensor& x, const std::vector<int64_t>& positions, KVCache& cache,
                       Tensor& out) const;

  // decode_step: void — one token x[B,1,D] at absolute position.
  void decode_step(const Tensor& x, int64_t position, KVCache& cache, Tensor& out) const;

  // config: const BlockConfig& — returns block hyperparameters.
  const BlockConfig& config() const { return config_; }

 private:
  BlockConfig config_;
  TransformerBlockWeights weights_;
};

}  // namespace serving
