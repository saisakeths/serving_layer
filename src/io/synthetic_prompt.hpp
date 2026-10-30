#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace serving {

// prompt_to_token_ids: vector<int32_t> — byte mod vocab; empty prompt → {0}.
inline std::vector<int32_t> prompt_to_token_ids(const std::string& prompt, int64_t vocab_size) {
  if (vocab_size <= 0) {
    throw std::invalid_argument("prompt_to_token_ids: vocab_size");
  }
  if (prompt.empty()) {
    return {0};
  }
  std::vector<int32_t> ids;
  ids.reserve(prompt.size());
  for (unsigned char c : prompt) {
    ids.push_back(static_cast<int32_t>(c % vocab_size));
  }
  return ids;
}

}  // namespace serving
