#include "model/llama/llama_cache.hpp"
#include "model/llama/llama_model.hpp"

#include <benchmark/benchmark.h>

using namespace std;

namespace serving {
namespace {

vector<int32_t> token_ids_for(int64_t seq, int64_t vocab) {
  vector<int32_t> ids(static_cast<size_t>(seq));
  for (int64_t i = 0; i < seq; ++i) {
    ids[static_cast<size_t>(i)] = static_cast<int32_t>(i % vocab);
  }
  return ids;
}

vector<int64_t> positions_for(int64_t seq) {
  vector<int64_t> pos(static_cast<size_t>(seq));
  for (int64_t i = 0; i < seq; ++i) {
    pos[static_cast<size_t>(i)] = i;
  }
  return pos;
}

}  // namespace
}  // namespace serving

static void BM_LlamaDecode_1B(benchmark::State& state) {
  using namespace serving;
  const int64_t prompt_len = 64;
  const int64_t decode_tokens = 64;
  LlamaConfig cfg = LlamaConfig::llama_1b_shape();
  LlamaWeights weights = make_random_llama_weights(cfg, 0.002f);
  LlamaModel model(cfg, move(weights));
  auto prompt_ids = token_ids_for(prompt_len, cfg.vocab_size);
  auto prompt_pos = positions_for(prompt_len);
  LlamaCache cache(cfg, 1, prompt_len + decode_tokens);
  Tensor prefill_logits({1, prompt_len, cfg.vocab_size});
  Tensor step_logits({1, 1, cfg.vocab_size});
  for (auto _ : state) {
    cache.reset();
    model.forward_prefill(prompt_ids, prompt_pos, cache, prefill_logits);
    const int32_t token = 0;
    for (int64_t t = 0; t < decode_tokens; ++t) {
      model.decode_step(token, prompt_len + t, cache, step_logits);
      benchmark::DoNotOptimize(step_logits.at({0, 0, 0}));
    }
  }
  state.SetItemsProcessed(state.iterations() * decode_tokens);
  state.counters["tokens/s"] =
      benchmark::Counter(static_cast<double>(state.iterations() * decode_tokens),
                         benchmark::Counter::kIsRate);
}

BENCHMARK(BM_LlamaDecode_1B)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
