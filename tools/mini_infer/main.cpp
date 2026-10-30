#include "io/synthetic_prompt.hpp"
#include "model/llama/llama_cache.hpp"
#include "model/llama/llama_model.hpp"
#include "sampler/generation_loop.hpp"
#include "sampler/sampler.hpp"

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

string parse_string_arg(int argc, char** argv, const string& name, const string& default_val) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == name) {
      return argv[i + 1];
    }
  }
  return default_val;
}

int parse_int_arg(int argc, char** argv, const string& name, int default_val) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == name) {
      return stoi(argv[i + 1]);
    }
  }
  return default_val;
}

uint64_t parse_u64_arg(int argc, char** argv, const string& name, uint64_t default_val) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == name) {
      return stoull(argv[i + 1]);
    }
  }
  return default_val;
}

float parse_float_arg(int argc, char** argv, const string& name, float default_val) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == name) {
      return stof(argv[i + 1]);
    }
  }
  return default_val;
}

bool synthetic_model_requested(int argc, char** argv) {
  if (parse_flag(argc, argv, "--synthetic")) {
    return true;
  }
  string model = parse_string_arg(argc, argv, "--model", "");
  return model == "synthetic";
}

void print_usage() {
  cerr << "Usage: mini_infer --model synthetic --prompt \"Hello\" --max-tokens 16 --greedy\n"
       << "       (--synthetic is an alias for --model synthetic)\n"
       << "  --config tiny|1b  --temperature F  --top-k N  --top-p F  --seed N\n";
}

}  // namespace

int main(int argc, char** argv) {
  using namespace serving;
  if (!synthetic_model_requested(argc, argv)) {
    print_usage();
    return 1;
  }

  string config_name = parse_string_arg(argc, argv, "--config", "tiny");
  string prompt = parse_string_arg(argc, argv, "--prompt", "Hello");
  int max_tokens = parse_int_arg(argc, argv, "--max-tokens", 16);
  bool greedy = parse_flag(argc, argv, "--greedy");

  LlamaConfig cfg = (config_name == "1b") ? LlamaConfig::llama_1b_shape() : LlamaConfig::tiny();
  if (max_tokens < 0) {
    cerr << "invalid --max-tokens\n";
    return 1;
  }

  vector<int32_t> prompt_ids = prompt_to_token_ids(prompt, cfg.vocab_size);
  const int64_t cache_len = static_cast<int64_t>(prompt_ids.size() + max_tokens);

  LlamaWeights weights = make_random_llama_weights(cfg, 0.01f);
  LlamaModel model(cfg, move(weights));
  LlamaCache cache(cfg, 1, cache_len);

  SamplerConfig scfg;
  scfg.greedy = greedy;
  scfg.temperature = parse_float_arg(argc, argv, "--temperature", 1.f);
  scfg.top_k = parse_int_arg(argc, argv, "--top-k", 0);
  scfg.top_p = parse_float_arg(argc, argv, "--top-p", 1.f);
  scfg.seed = parse_u64_arg(argc, argv, "--seed", 0);
  Sampler sampler(scfg);

  GenerationConfig gcfg;
  gcfg.max_tokens = max_tokens;
  gcfg.eos_token_id = cfg.eos_token_id;

  vector<int32_t> generated;
  try {
    generated = generate(model, cache, prompt_ids, sampler, gcfg);
  } catch (const exception& e) {
    cerr << "generate failed: " << e.what() << "\n";
    return 1;
  }

  cout << "prompt_tokens: ";
  for (int32_t id : prompt_ids) {
    cout << id << " ";
  }
  cout << "\ngenerated_tokens: ";
  for (int32_t id : generated) {
    cout << id << " ";
  }
  cout << "\n";
  if (generated.empty()) {
    cerr << "warning: empty generation\n";
    return 1;
  }
  return 0;
}
