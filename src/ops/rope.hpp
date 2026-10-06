#pragma once

#include "core/tensor.hpp"

#include <cstdint>
#include <vector>

namespace serving {

// apply_rope: void — applies Llama-style RoPE to q and k [batch, seq, n_heads, head_dim].
void apply_rope(Tensor& q, Tensor& k, const std::vector<int64_t>& positions, float rope_theta);

}  // namespace serving
