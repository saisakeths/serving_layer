#include "sampler/generation_loop.hpp"

#include <stdexcept>

using namespace std;

namespace serving {

// generate: vector<int32_t> — autoregressive decode; returns newly generated token ids only.
vector<int32_t> generate(const LlamaModel& model, LlamaCache& cache,
                         const vector<int32_t>& prompt_ids, const Sampler& sampler,
                         GenerationConfig cfg) {
  if (cfg.max_tokens < 0) {
    throw invalid_argument("generate: max_tokens");
  }
  const int64_t prompt_len = static_cast<int64_t>(prompt_ids.size());
  if (prompt_len == 0) {
    throw invalid_argument("generate: empty prompt");
  }
  if (prompt_len + cfg.max_tokens > cache.max_seq()) {
    throw invalid_argument("generate: prompt + max_tokens exceeds cache");
  }

  vector<int64_t> positions(static_cast<size_t>(prompt_len));
  for (int64_t i = 0; i < prompt_len; ++i) {
    positions[static_cast<size_t>(i)] = i;
  }

  const int64_t vocab = model.config().vocab_size;
  Tensor prefill_logits({1, prompt_len, vocab});
  model.forward_prefill(prompt_ids, positions, cache, prefill_logits);

  vector<int32_t> generated;
  int64_t position = prompt_len - 1;
  Tensor step_logits({1, 1, vocab});

  for (int t = 0; t < cfg.max_tokens; ++t) {
    int32_t next = 0;
    if (t == 0) {
      Tensor last_row({1, 1, vocab});
      for (int64_t v = 0; v < vocab; ++v) {
        last_row.at({0, 0, v}) = prefill_logits.at({0, prompt_len - 1, v});
      }
      next = sampler.sample(last_row);
    } else {
      next = sampler.sample(step_logits);
    }
    generated.push_back(next);
    if (cfg.eos_token_id >= 0 && next == cfg.eos_token_id) {
      break;
    }
    position = prompt_len + t;
    model.decode_step(next, position, cache, step_logits);
  }
  return generated;
}

}  // namespace serving
