#pragma once

#include "cache/kv_cache.hpp"
#include "model/llama/llama_config.hpp"

#include <cstdint>
#include <vector>

namespace serving {

// LlamaCache: — one KVCache per transformer layer for prefill and decode.
class LlamaCache {
 public:
  // LlamaCache: — allocates n_layer KV caches with shared batch and max_seq.
  LlamaCache(const LlamaConfig& config, int64_t batch, int64_t max_seq);

  // layer: KVCache& — KV storage for layer index.
  KVCache& layer(size_t layer_index);
  // layer: const KVCache& — KV storage for layer index.
  const KVCache& layer(size_t layer_index) const;

  // max_seq: int64_t — sequence capacity of each layer cache.
  int64_t max_seq() const;

 private:
  std::vector<KVCache> layers_;
};

}  // namespace serving
