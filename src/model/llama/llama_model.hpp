#pragma once

#include "core/tensor.hpp"
#include "model/llama/llama_cache.hpp"
#include "model/llama/llama_config.hpp"
#include "model/llama/llama_weights.hpp"
#include "model/llama/transformer_block.hpp"

#include <cstdint>
#include <vector>

namespace serving {

class LlamaModel {
 public:
  // LlamaModel: — stacks transformer blocks and output head from config and weights.
  LlamaModel(LlamaConfig config, LlamaWeights weights);

  // forward: void — prefill with KV cache; logits shape [1, seq, vocab_size].
  void forward(const std::vector<int32_t>& input_ids, const std::vector<int64_t>& positions,
               LlamaCache& cache, Tensor& logits) const;

  // config: const LlamaConfig& — model hyperparameters.
  const LlamaConfig& config() const { return config_; }

 private:
  const Tensor& head_weight() const;

  LlamaConfig config_;
  LlamaWeights weights_;
  std::vector<TransformerBlock> blocks_;
};

}  // namespace serving
