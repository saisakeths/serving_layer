#include "model/llama/llama_cache.hpp"

#include <stdexcept>

using namespace std;

namespace serving {

// LlamaCache: — allocates n_layer KV caches with shared batch and max_seq.
LlamaCache::LlamaCache(const LlamaConfig& config, int64_t batch, int64_t max_seq)
    : layers_() {
  config.validate();
  if (batch <= 0 || max_seq <= 0) {
    throw invalid_argument("LlamaCache: batch and max_seq must be positive");
  }
  layers_.reserve(static_cast<size_t>(config.n_layer));
  for (int64_t i = 0; i < config.n_layer; ++i) {
    layers_.emplace_back(batch, max_seq, config.n_kv_heads, config.head_dim);
  }
}

// layer: KVCache& — KV storage for layer index.
KVCache& LlamaCache::layer(size_t layer_index) {
  if (layer_index >= layers_.size()) {
    throw out_of_range("LlamaCache layer index");
  }
  return layers_[layer_index];
}

// layer: const KVCache& — KV storage for layer index.
const KVCache& LlamaCache::layer(size_t layer_index) const {
  if (layer_index >= layers_.size()) {
    throw out_of_range("LlamaCache layer index");
  }
  return layers_[layer_index];
}

// max_seq: int64_t — sequence capacity of each layer cache.
int64_t LlamaCache::max_seq() const {
  if (layers_.empty()) {
    return 0;
  }
  return layers_.front().max_seq();
}

}  // namespace serving
