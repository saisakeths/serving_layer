#include "cache/kv_cache.hpp"

#include <stdexcept>

using namespace std;

namespace serving {

// KVCache: — allocates fixed K/V buffers for batch and max_seq.
KVCache::KVCache(int64_t batch, int64_t max_seq, int64_t n_kv_heads, int64_t head_dim)
    : batch_(batch),
      max_seq_(max_seq),
      n_kv_heads_(n_kv_heads),
      head_dim_(head_dim),
      k_({batch, max_seq, n_kv_heads, head_dim}),
      v_({batch, max_seq, n_kv_heads, head_dim}) {
  if (batch <= 0 || max_seq <= 0 || n_kv_heads <= 0 || head_dim <= 0) {
    throw invalid_argument("KVCache dimensions must be positive");
  }
}

// reset: void — clears valid length (does not zero buffer).
void KVCache::reset() { seq_len_ = 0; }

// append: void — copies k,v [batch, seq_new, n_kv_heads, head_dim] after current seq_len.
void KVCache::append(const Tensor& k, const Tensor& v) {
  const auto& ks = k.shape();
  const auto& vs = v.shape();
  if (ks != vs) {
    throw invalid_argument("KVCache append k/v shape mismatch");
  }
  if (ks.size() != 4 || ks[0] != batch_ || ks[2] != n_kv_heads_ || ks[3] != head_dim_) {
    throw invalid_argument("KVCache append shape");
  }
  int64_t seq_new = ks[1];
  if (seq_len_ + seq_new > max_seq_) {
    throw invalid_argument("KVCache capacity exceeded");
  }
  for (int64_t b = 0; b < batch_; ++b) {
    for (int64_t s = 0; s < seq_new; ++s) {
      int64_t t = seq_len_ + s;
      for (int64_t h = 0; h < n_kv_heads_; ++h) {
        for (int64_t d = 0; d < head_dim_; ++d) {
          k_.at({b, t, h, d}) = k.at({b, s, h, d});
          v_.at({b, t, h, d}) = v.at({b, s, h, d});
        }
      }
    }
  }
  seq_len_ += seq_new;
}

}  // namespace serving
