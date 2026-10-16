#pragma once

#include "core/tensor.hpp"
#include "model/llama/llama_config.hpp"
#include "model/llama/transformer_block.hpp"

#include <cstdint>
#include <vector>

namespace serving {

struct LlamaWeights {
  Tensor tok_embeddings;
  std::vector<TransformerBlockWeights> layers;
  Tensor output_norm;
  Tensor lm_head;
};

// validate_llama_weights: void — checks tensor shapes against config.
void validate_llama_weights(const LlamaConfig& config, const LlamaWeights& weights);

// make_random_llama_weights: LlamaWeights — deterministic random weights for tests and benches.
LlamaWeights make_random_llama_weights(const LlamaConfig& config, float scale);

}  // namespace serving
