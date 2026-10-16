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

static void BM_LlamaForward_1B(benchmark::State& state) {
  using namespace serving;
  int64_t seq = state.range(0);
  LlamaConfig cfg = LlamaConfig::llama_1b_shape();
  LlamaWeights weights = make_random_llama_weights(cfg, 0.002f);
  LlamaModel model(cfg, move(weights));
  auto ids = token_ids_for(seq, cfg.vocab_size);
  auto pos = positions_for(seq);
  LlamaCache cache(cfg, 1, seq);
  Tensor logits({1, seq, cfg.vocab_size});
  for (auto _ : state) {
    model.forward(ids, pos, cache, logits);
    benchmark::DoNotOptimize(logits.at({0, seq - 1, 0}));
  }
  state.SetItemsProcessed(state.iterations() * seq);
  state.counters["tokens/s"] =
      benchmark::Counter(static_cast<double>(state.iterations() * seq),
                         benchmark::Counter::kIsRate);
}

BENCHMARK(BM_LlamaForward_1B)->
                        Arg(64)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
