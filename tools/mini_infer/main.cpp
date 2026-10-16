#include "model/llama/llama_cache.hpp"
#include "model/llama/llama_model.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

using namespace std;

namespace {

bool parse_flag(int argc, char** argv, const string& name) {
  for (int i = 1; i < argc; ++i) {
    if (argv[i] == name) {
      return true;
    }
  }
  return false;
}

int parse_int_arg(int argc, char** argv, const string& name, int default_val) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == name) {
      return stoi(argv[i + 1]);
    }
  }
  return default_val;
}

int32_t argmax_row(const serving::Tensor& logits, int64_t row) {
  int64_t vocab = logits.shape()[2];
  int32_t best = 0;
  float best_v = logits.at({0, row, 0});
  for (int64_t v = 1; v < vocab; ++v) {
    float x = logits.at({0, row, v});
    if (x > best_v) {
      best_v = x;
      best = static_cast<int32_t>(v);
    }
  }
  return best;
}

}  // namespace

int main(int argc, char** argv) {
  using namespace serving;
  if (!parse_flag(argc, argv, "--synthetic")) {
    cerr << "Usage: mini_infer --synthetic [--seq N] [--config tiny|1b]\n";
    return 1;
  }
  string config_name = "tiny";
  for (int i = 1; i + 1 < argc; ++i) {
    if (string(argv[i]) == "--config") {
      config_name = argv[i + 1];
    }
  }
  int seq = parse_int_arg(argc, argv, "--seq", 8);

  LlamaConfig cfg = (config_name == "1b") ? LlamaConfig::llama_1b_shape() : LlamaConfig::tiny();
  if (seq <= 0 || seq > cfg.vocab_size) {
    cerr << "invalid --seq\n";
    return 1;
  }

  LlamaWeights weights = make_random_llama_weights(cfg, 0.01f);
  LlamaModel model(cfg, move(weights));

  vector<int32_t> ids(static_cast<size_t>(seq));
  vector<int64_t> pos(static_cast<size_t>(seq));
  for (int i = 0; i < seq; ++i) {
    ids[static_cast<size_t>(i)] = static_cast<int32_t>(i % cfg.vocab_size);
    pos[static_cast<size_t>(i)] = i;
  }

  LlamaCache cache(cfg, 1, seq);
  Tensor logits({1, seq, cfg.vocab_size});
  model.forward(ids, pos, cache, logits);

  cout << "logits shape: [1, " << seq << ", " << cfg.vocab_size << "]\n";
  cout << "last_token_argmax: " << argmax_row(logits, seq - 1) << "\n";
  return 0;
}
