#include "model/llama/llama_weights.hpp"

#include <stdexcept>

using namespace std;

namespace serving {
namespace {

// fill_tensor_pattern: void — fills tensor with a deterministic pattern.
void fill_tensor_pattern(Tensor& t, float scale) {
  for (size_t i = 0; i < t.numel(); ++i) {
    t.data()[i] = scale * static_cast<float>(static_cast<int>(i % 97) - 48);
  }
}

// make_random_block_weights: TransformerBlockWeights — one layer with scaled random values.
TransformerBlockWeights make_random_block_weights(const BlockConfig& cfg, float scale) {
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
  fill_tensor_pattern(w.attn.wq, scale);
  fill_tensor_pattern(w.attn.wk, scale);
  fill_tensor_pattern(w.attn.wv, scale);
  fill_tensor_pattern(w.attn.wo, scale);
  fill_tensor_pattern(w.w_gate, scale);
  fill_tensor_pattern(w.w_up, scale);
  fill_tensor_pattern(w.w_down, scale);
  return w;
}

}  // namespace

// validate_llama_weights: void — checks tensor shapes against config.
void validate_llama_weights(const LlamaConfig& config, const LlamaWeights& weights) {
  config.validate();
  const int64_t v = config.vocab_size;
  const int64_t d = config.dim;
  if (weights.tok_embeddings.shape() != vector<int64_t>{v, d}) {
    throw invalid_argument("tok_embeddings shape");
  }
  if (static_cast<int64_t>(weights.layers.size()) != config.n_layer) {
    throw invalid_argument("layers count");
  }
  if (weights.output_norm.shape() != vector<int64_t>{d}) {
    throw invalid_argument("output_norm shape");
  }
  if (!config.tie_word_embeddings) {
    if (weights.lm_head.shape() != vector<int64_t>{v, d}) {
      throw invalid_argument("lm_head shape");
    }
  }
}

// make_random_llama_weights: LlamaWeights — deterministic random weights for tests and benches.
LlamaWeights make_random_llama_weights(const LlamaConfig& config, float scale) {
  config.validate();
  LlamaWeights w;
  w.tok_embeddings = Tensor({config.vocab_size, config.dim});
  fill_tensor_pattern(w.tok_embeddings, scale);
  w.output_norm = Tensor({config.dim});
  for (int64_t i = 0; i < config.dim; ++i) {
    w.output_norm.at({i}) = 1.f;
  }
  BlockConfig bc = config.block_config();
  w.layers.reserve(static_cast<size_t>(config.n_layer));
  for (int64_t layer = 0; layer < config.n_layer; ++layer) {
    w.layers.push_back(make_random_block_weights(bc, scale));
  }
  if (!config.tie_word_embeddings) {
    w.lm_head = Tensor({config.vocab_size, config.dim});
    fill_tensor_pattern(w.lm_head, scale);
  }
  validate_llama_weights(config, w);
  return w;
}

}  // namespace serving
