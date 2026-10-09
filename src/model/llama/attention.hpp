#pragma once

#include "cache/kv_cache.hpp"
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

// gqa_attention_prefill: void — same as gqa_attention; appends post-RoPE K/V into cache.
void gqa_attention_prefill(const Tensor& x, const AttentionWeights& weights,
                             const BlockConfig& config, const std::vector<int64_t>& positions,
                             KVCache& cache, Tensor& out);

// gqa_attention_decode: void — one token x[B,1,D]; uses cache and absolute position.
void gqa_attention_decode(const Tensor& x, const AttentionWeights& weights,
                          const BlockConfig& config, int64_t position, KVCache& cache,
                          Tensor& out);

}  // namespace serving
