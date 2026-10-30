#pragma once

#include "model/llama/llama_cache.hpp"
#include "model/llama/llama_model.hpp"
#include "sampler/sampler.hpp"

#include <cstdint>
#include <vector>

namespace serving {

struct GenerationConfig {
  int max_tokens = 16;
  // eos_token_id < 0 disables early stop on EOS.
  int32_t eos_token_id = 0;
};

// generate: vector<int32_t> — autoregressive decode; returns newly generated token ids only.
std::vector<int32_t> generate(const LlamaModel& model, LlamaCache& cache,
                              const std::vector<int32_t>& prompt_ids, const Sampler& sampler,
                              GenerationConfig cfg);

}  // namespace serving
