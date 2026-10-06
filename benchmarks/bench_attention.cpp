#include "model/llama/transformer_block.hpp"

#include <benchmark/benchmark.h>

using namespace std;

namespace serving {
namespace {

BlockConfig config_1b_shape() {
  BlockConfig c;
  c.dim = 2048;
  c.n_heads = 32;
  c.n_kv_heads = 8;
  c.head_dim = 64;
  c.ffn_dim = 5632;
  c.rope_theta = 10000.f;
  return c;
}

TransformerBlockWeights make_random_weights(const BlockConfig& cfg, float scale) {
  TransformerBlockWeights w;
  w.attn_norm = Tensor({cfg.dim});
  w.ffn_norm = Tensor({cfg.dim});
  for (int64_t i = 0; i < cfg.dim; ++i) {
    w.attn_norm.at({i}) = 1.f;
    w.ffn_norm.at({i}) = 1.f;
  }
  w.attn.wq = Tensor({cfg.dim, cfg.dim});
  w.attn.wk = Tensor({cfg.n_kv_heads * cfg.head_dim, cfg.dim});
  w.attn.wv = Tensor({cfg.n_kv_heads * cfg.head_dim, cfg.dim});
  w.attn.wo = Tensor({cfg.dim, cfg.dim});
  w.w_gate = Tensor({cfg.ffn_dim, cfg.dim});
  w.w_up = Tensor({cfg.ffn_dim, cfg.dim});
  w.w_down = Tensor({cfg.dim, cfg.ffn_dim});
  auto fill = [&](Tensor& t) {
    for (size_t i = 0; i < t.numel(); ++i) {
      t.data()[i] = scale * static_cast<float>(static_cast<int>(i % 97) - 48);
    }
  };
  fill(w.attn.wq);
  fill(w.attn.wk);
  fill(w.attn.wv);
  fill(w.attn.wo);
  fill(w.w_gate);
  fill(w.w_up);
  fill(w.w_down);
  return w;
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

static void BM_Block_Prefill(benchmark::State& state) {
  using namespace serving;
  int64_t seq = state.range(0);
  BlockConfig cfg = config_1b_shape();
  TransformerBlock block(cfg, make_random_weights(cfg, 0.002f));
  Tensor x({1, seq, cfg.dim});
  for (size_t i = 0; i < x.numel(); ++i) {
    x.data()[i] = 0.01f * static_cast<float>(static_cast<int>(i % 53));
  }
  auto pos = positions_for(seq);
  Tensor out({1, seq, cfg.dim});
  for (auto _ : state) {
    block.forward(x, pos, out);
    benchmark::DoNotOptimize(out.data());
  }
  state.SetItemsProcessed(state.iterations() * seq);
  state.counters["tokens/s"] =
      benchmark::Counter(static_cast<double>(state.iterations() * seq),
                         benchmark::Counter::kIsRate);
}
BENCHMARK(BM_Block_Prefill)->Arg(128)->Arg(512)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
