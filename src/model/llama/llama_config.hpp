#pragma once

#include "model/llama/block_config.hpp"

#include <cstdint>
#include <stdexcept>

namespace serving {

struct LlamaConfig {
  int64_t dim = 0;
  int64_t n_heads = 0;
  int64_t n_kv_heads = 0;
  int64_t head_dim = 0;
  int64_t ffn_dim = 0;
  int64_t n_layer = 0;
  int64_t vocab_size = 0;
  float rope_theta = 10000.f;
  float rms_eps = 1e-5f;
  bool tie_word_embeddings = true;

  // validate: void — throws if hyperparameters are inconsistent.
  void validate() const {
    if (dim <= 0 || n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || ffn_dim <= 0 ||
        n_layer <= 0 || vocab_size <= 0) {
      throw std::invalid_argument("LlamaConfig: invalid dimensions");
    }
    if (n_heads * head_dim != dim) {
      throw std::invalid_argument("LlamaConfig: n_heads * head_dim must equal dim");
    }
    if (n_kv_heads > n_heads || n_heads % n_kv_heads != 0) {
      throw std::invalid_argument("LlamaConfig: invalid GQA head counts");
    }
  }

  // block_config: BlockConfig — per-layer block hyperparameters.
  BlockConfig block_config() const {
    BlockConfig c;
    c.dim = dim;
    c.n_heads = n_heads;
    c.n_kv_heads = n_kv_heads;
    c.head_dim = head_dim;
    c.ffn_dim = ffn_dim;
    c.rope_theta = rope_theta;
    c.rms_eps = rms_eps;
    return c;
  }

  // tiny: LlamaConfig — small config for unit tests.
  static LlamaConfig tiny() {
    LlamaConfig c;
    c.dim = 8;
    c.n_heads = 2;
    c.n_kv_heads = 1;
    c.head_dim = 4;
    c.ffn_dim = 16;
    c.n_layer = 2;
    c.vocab_size = 32;
    c.rope_theta = 10000.f;
    c.tie_word_embeddings = true;
    return c;
  }

  // llama_1b_shape: LlamaConfig — Llama-3.2-1B-class geometry for cloud benchmarks.
  static LlamaConfig llama_1b_shape() {
    LlamaConfig c;
    c.dim = 2048;
    c.n_heads = 32;
    c.n_kv_heads = 8;
    c.head_dim = 64;
    c.ffn_dim = 5632;
    c.n_layer = 16;
    c.vocab_size = 128256;
    c.rope_theta = 10000.f;
    c.tie_word_embeddings = true;
    return c;
  }
};

}  // namespace serving
