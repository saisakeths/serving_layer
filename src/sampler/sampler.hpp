#pragma once

#include "core/tensor.hpp"

#include <cstdint>
#include <random>
#include <vector>

using namespace std;

namespace serving {

struct SamplerConfig {
  bool greedy = false;
  float temperature = 1.f;
  int top_k = 0;
  float top_p = 1.f;
  uint64_t seed = 0;
};

class Sampler {
 public:
  // Sampler: — constructs sampler from config (validates parameters).
  explicit Sampler(SamplerConfig config);

  // sample: int32_t — picks next token from logits [1, 1, vocab] or [1, vocab].
  int32_t sample(const Tensor& logits) const;

 private:
  // logits_vocab_row: vector<float> — copies vocab logits from [1,1,V] or [1,V].
  static vector<float> logits_vocab_row(const Tensor& logits);

  // apply_temperature: void — divides logits by temperature when T > 0.
  static void apply_temperature(vector<float>& logits, float temperature);

  // apply_top_k: void — masks logits below the k-th largest value.
  static void apply_top_k(vector<float>& logits, int top_k);

  // softmax_probs: vector<float> — stable softmax over logits.
  static vector<float> softmax_probs(const vector<float>& logits);

  // apply_top_p: vector<float> — nucleus filter; returns renormalized probs.
  static vector<float> apply_top_p(vector<float> probs, float top_p);

  // greedy_pick: int32_t — argmax over logits.
  static int32_t greedy_pick(const vector<float>& logits);

  SamplerConfig config_;
  mutable mt19937 rng_;
};

}  // namespace serving
