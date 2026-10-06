#pragma once

#include "core/tensor.hpp"
#include "model/llama/block_config.hpp"

#include <vector>

namespace serving {

struct AttentionWeights {
  Tensor wq;
  Tensor wk;
  Tensor wv;
  Tensor wo;
};

// gqa_attention: void — causal GQA attention; x[B,S,D] -> out[B,S,D] (full sequence, no KV cache).
void gqa_attention(const Tensor& x, const AttentionWeights& weights, const BlockConfig& config,
                   const std::vector<int64_t>& positions, Tensor& out);

}  // namespace serving
