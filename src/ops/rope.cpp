#include "ops/rope.hpp"

#include <cmath>
#include <stdexcept>

using namespace std;

namespace serving {

namespace {

// rotate_half_pair: void — applies one frequency pair rotation at (head, dim_pair).
void rotate_half_pair(Tensor& t, int64_t b, int64_t s, int64_t h, int64_t pair, float cos_v,
                      float sin_v) {
  int64_t d0 = pair * 2;
  int64_t d1 = d0 + 1;
  float x0 = t.at({b, s, h, d0});
  float x1 = t.at({b, s, h, d1});
  t.at({b, s, h, d0}) = x0 * cos_v - x1 * sin_v;
  t.at({b, s, h, d1}) = x0 * sin_v + x1 * cos_v;
}

}  // namespace

// apply_rope: void — applies Llama-style RoPE to q and k [batch, seq, n_heads, head_dim].
void apply_rope(Tensor& q, Tensor& k, const vector<int64_t>& positions, float rope_theta) {
  const auto& qs = q.shape();
  const auto& ks = k.shape();
  if (qs.size() != 4 || ks.size() != 4) {
    throw invalid_argument("rope expects q,k [batch, seq, heads, head_dim]");
  }
  if (qs[0] != ks[0] || qs[1] != ks[1] || qs[3] != ks[3]) {
    throw invalid_argument("rope q,k batch/seq/head_dim mismatch");
  }
  int64_t batch = qs[0];
  int64_t seq = qs[1];
  int64_t n_heads = qs[2];
  int64_t head_dim = qs[3];
  if (static_cast<int64_t>(positions.size()) != seq) {
    throw invalid_argument("positions length must match seq");
  }
  if (head_dim % 2 != 0) {
    throw invalid_argument("head_dim must be even");
  }
  int64_t n_pairs = head_dim / 2;
  vector<float> cos_cache(static_cast<size_t>(seq * n_pairs));
  vector<float> sin_cache(static_cast<size_t>(seq * n_pairs));
  for (int64_t s = 0; s < seq; ++s) {
    float pos = static_cast<float>(positions[s]);
    for (int64_t p = 0; p < n_pairs; ++p) {
      float inv_freq = powf(rope_theta, -2.f * static_cast<float>(p) / static_cast<float>(head_dim));
      float angle = pos * inv_freq;
      cos_cache[static_cast<size_t>(s * n_pairs + p)] = cosf(angle);
      sin_cache[static_cast<size_t>(s * n_pairs + p)] = sinf(angle);
    }
  }
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t s = 0; s < seq; ++s) {
      for (int64_t h = 0; h < n_heads; ++h) {
        for (int64_t p = 0; p < n_pairs; ++p) {
          float cos_v = cos_cache[static_cast<size_t>(s * n_pairs + p)];
          float sin_v = sin_cache[static_cast<size_t>(s * n_pairs + p)];
          rotate_half_pair(q, b, s, h, p, cos_v, sin_v);
        }
      }
    }
  }
  int64_t n_kv_heads = ks[2];
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t s = 0; s < seq; ++s) {
      for (int64_t h = 0; h < n_kv_heads; ++h) {
        for (int64_t p = 0; p < n_pairs; ++p) {
          float cos_v = cos_cache[static_cast<size_t>(s * n_pairs + p)];
          float sin_v = sin_cache[static_cast<size_t>(s * n_pairs + p)];
          rotate_half_pair(k, b, s, h, p, cos_v, sin_v);
        }
      }
    }
  }
}

}  // namespace serving
