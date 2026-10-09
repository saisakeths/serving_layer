#pragma once

#include "core/tensor.hpp"

#include <cstdint>

namespace serving {

// KVCache: — contiguous post-RoPE K/V storage [batch, max_seq, n_kv_heads, head_dim].
class KVCache {
 public:
  // KVCache: — allocates fixed K/V buffers for batch and max_seq.
  KVCache(int64_t batch, int64_t max_seq, int64_t n_kv_heads, int64_t head_dim);

  // seq_len: int64_t — number of timesteps written.
  int64_t seq_len() const { return seq_len_; }
  // max_seq: int64_t — capacity along sequence axis.
  int64_t max_seq() const { return max_seq_; }
  // batch: int64_t — batch size.
  int64_t batch() const { return batch_; }
  // n_kv_heads: int64_t — number of KV heads.
  int64_t n_kv_heads() const { return n_kv_heads_; }
  // head_dim: int64_t — head dimension.
  int64_t head_dim() const { return head_dim_; }

  // reset: void — clears valid length (does not zero buffer).
  void reset();

  // append: void — copies k,v [batch, seq_new, n_kv_heads, head_dim] after current seq_len.
  void append(const Tensor& k, const Tensor& v);

  // keys: const Tensor& — full K buffer; index time t in [0, seq_len).
  const Tensor& keys() const { return k_; }
  // values: const Tensor& — full V buffer; index time t in [0, seq_len).
  const Tensor& values() const { return v_; }

 private:
  int64_t batch_;
  int64_t max_seq_;
  int64_t n_kv_heads_;
  int64_t head_dim_;
  int64_t seq_len_ = 0;
  Tensor k_;
  Tensor v_;
};

}  // namespace serving
