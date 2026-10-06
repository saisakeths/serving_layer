#pragma once

#include <cstdint>

namespace serving {

struct BlockConfig {
  int64_t dim = 0;
  int64_t n_heads = 0;
  int64_t n_kv_heads = 0;
  int64_t head_dim = 0;
  int64_t ffn_dim = 0;
  float rope_theta = 10000.f;
  float rms_eps = 1e-5f;
};

}  // namespace serving
