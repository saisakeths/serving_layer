#include "cache/kv_cache.hpp"
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

static void BM_Block_DecodeNaive(benchmark::State& state) {
  using namespace serving;
  int64_t prompt_len = state.range(0);
  int64_t decode_tokens = state.range(1);
  BlockConfig cfg = config_1b_shape();
  TransformerBlock block(cfg, make_random_weights(cfg, 0.002f));
  Tensor prompt({1, prompt_len, cfg.dim});
  for (size_t i = 0; i < prompt.numel(); ++i) {
    prompt.data()[i] = 0.01f * static_cast<float>(static_cast<int>(i % 53));
  }
  Tensor next_tok({1, 1, cfg.dim});
  for (size_t i = 0; i < next_tok.numel(); ++i) {
    next_tok.data()[i] = 0.02f * static_cast<float>(static_cast<int>(i % 41));
  }
  for (auto _ : state) {
    for (int64_t t = 0; t < decode_tokens; ++t) {
      int64_t cur_len = prompt_len + t + 1;
      Tensor x({1, cur_len, cfg.dim});
      Tensor out({1, cur_len, cfg.dim});
      for (int64_t s = 0; s < prompt_len; ++s) {
        for (int64_t d = 0; d < cfg.dim; ++d) {
          x.at({0, s, d}) = prompt.at({0, s, d});
        }
      }
      for (int64_t s = prompt_len; s < cur_len; ++s) {
        for (int64_t d = 0; d < cfg.dim; ++d) {
          x.at({0, s, d}) = next_tok.at({0, 0, d});
        }
      }
      block.forward(x, positions_for(cur_len), out);
      benchmark::DoNotOptimize(out.at({0, cur_len - 1, 0}));
    }
  }
  state.SetItemsProcessed(state.iterations() * decode_tokens);
  state.counters["tokens/s"] =
      benchmark::Counter(static_cast<double>(state.iterations() * decode_tokens),
                         benchmark::Counter::kIsRate);
}
BENCHMARK(BM_Block_DecodeNaive)
    ->Args({64, 8})
    ->Args({128, 8})
    ->Args({256, 8})
    ->Unit(benchmark::kMillisecond);

static void BM_Block_DecodeKV(benchmark::State& state) {
  using namespace serving;
  int64_t prompt_len = state.range(0);
  int64_t decode_tokens = state.range(1);
  BlockConfig cfg = config_1b_shape();
  TransformerBlock block(cfg, make_random_weights(cfg, 0.002f));
  Tensor prompt({1, prompt_len, cfg.dim});
  for (size_t i = 0; i < prompt.numel(); ++i) {
    prompt.data()[i] = 0.01f * static_cast<float>(static_cast<int>(i % 53));
  }
  Tensor next_tok({1, 1, cfg.dim});
  for (size_t i = 0; i < next_tok.numel(); ++i) {
    next_tok.data()[i] = 0.02f * static_cast<float>(static_cast<int>(i % 41));
  }
  KVCache cache(1, prompt_len + decode_tokens, cfg.n_kv_heads, cfg.head_dim);
  Tensor out_prefill({1, prompt_len, cfg.dim});
  Tensor out_step({1, 1, cfg.dim});
  for (auto _ : state) {
    cache.reset();
    block.forward_prefill(prompt, positions_for(prompt_len), cache, out_prefill);
    for (int64_t t = 0; t < decode_tokens; ++t) {
      block.decode_step(next_tok, prompt_len + t, cache, out_step);
      benchmark::DoNotOptimize(out_step.at({0, 0, 0}));
    }
  }
  state.SetItemsProcessed(state.iterations() * decode_tokens);
  state.counters["tokens/s"] =
      benchmark::Counter(static_cast<double>(state.iterations() * decode_tokens),
                         benchmark::Counter::kIsRate);
}
BENCHMARK(BM_Block_DecodeKV)
    ->Args({64, 8})
    ->Args({128, 8})
    ->Args({256, 8})
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
