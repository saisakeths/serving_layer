#include "sampler/sampler.hpp"

#include "ops/ops.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

using namespace std;

namespace serving {

// logits_vocab_row: vector<float> — copies vocab logits from [1,1,V] or [1,V].
vector<float> Sampler::logits_vocab_row(const Tensor& logits) {
  const auto& sh = logits.shape();
  int64_t vocab = 0;
  if (sh.size() == 3 && sh[0] == 1 && sh[1] == 1) {
    vocab = sh[2];
    vector<float> row(static_cast<size_t>(vocab));
    for (int64_t v = 0; v < vocab; ++v) {
      row[static_cast<size_t>(v)] = logits.at({0, 0, v});
    }
    return row;
  }
  if (sh.size() == 2 && sh[0] == 1) {
    vocab = sh[1];
    vector<float> row(static_cast<size_t>(vocab));
    for (int64_t v = 0; v < vocab; ++v) {
      row[static_cast<size_t>(v)] = logits.at({0, v});
    }
    return row;
  }
  throw invalid_argument("Sampler: logits shape");
}

// apply_temperature: void — divides logits by temperature when T > 0.
void Sampler::apply_temperature(vector<float>& logits, float temperature) {
  if (temperature <= 0.f) {
    throw invalid_argument("Sampler: temperature must be positive");
  }
  if (fabs(temperature - 1.f) < 1e-6f) {
    return;
  }
  for (float& x : logits) {
    x /= temperature;
  }
}

// apply_top_k: void — masks logits below the k-th largest value.
void Sampler::apply_top_k(vector<float>& logits, int top_k) {
  if (top_k <= 0 || top_k >= static_cast<int>(logits.size())) {
    return;
  }
  vector<float> sorted = logits;
  nth_element(sorted.begin(), sorted.begin() + (top_k - 1), sorted.end(), greater<float>());
  float threshold = sorted[top_k - 1];
  for (float& x : logits) {
    if (x < threshold) {
      x = -1e30f;
    }
  }
}

// softmax_probs: vector<float> — stable softmax over logits.
vector<float> Sampler::softmax_probs(const vector<float>& logits) {
  Tensor in({1, static_cast<int64_t>(logits.size())});
  Tensor out({1, static_cast<int64_t>(logits.size())});
  for (size_t i = 0; i < logits.size(); ++i) {
    in.at({0, static_cast<int64_t>(i)}) = logits[i];
  }
  softmax_last_dim(in, out);
  vector<float> probs(logits.size());
  for (size_t i = 0; i < logits.size(); ++i) {
    probs[i] = out.at({0, static_cast<int64_t>(i)});
  }
  return probs;
}

// apply_top_p: vector<float> — nucleus filter; returns renormalized probs.
vector<float> Sampler::apply_top_p(vector<float> probs, float top_p) {
  if (top_p >= 1.f) {
    return probs;
  }
  if (top_p <= 0.f) {
    throw invalid_argument("Sampler: top_p must be in (0, 1]");
  }
  size_t vocab_size = probs.size();
  vector<size_t> order(vocab_size);
  iota(order.begin(), order.end(), 0);
  sort(order.begin(), order.end(),
       [&](size_t a, size_t b) { return probs[a] > probs[b]; });
  float cum = 0.f;
  size_t keep = 0;
  for (size_t i = 0; i < vocab_size; ++i) {
    cum += probs[order[i]];
    keep = i + 1;
    if (cum >= top_p) {
      break;
    }
  }
  if (keep == 0) {
    keep = 1;
  }
  vector<float> filtered(vocab_size, 0.f);
  float sum = 0.f;
  for (size_t i = 0; i < keep; ++i) {
    size_t idx = order[i];
    filtered[idx] = probs[idx];
    sum += probs[idx];
  }
  if (sum <= 0.f) {
    throw runtime_error("Sampler: top_p removed all mass");
  }
  for (float& p : filtered) {
    p /= sum;
  }
  return filtered;
}

// greedy_pick: int32_t — argmax over logits.
int32_t Sampler::greedy_pick(const vector<float>& logits) {
  int32_t best = 0;
  float best_v = logits[0];
  for (size_t i = 1; i < logits.size(); ++i) {
    if (logits[i] > best_v) {
      best_v = logits[i];
      best = static_cast<int32_t>(i);
    }
  }
  return best;
}

// Sampler: — constructs sampler from config (validates parameters).
Sampler::Sampler(SamplerConfig config) : config_(config), rng_(static_cast<uint32_t>(config.seed)) {
  if (config_.temperature <= 0.f) {
    throw invalid_argument("Sampler: temperature must be positive");
  }
  if (config_.top_p <= 0.f || config_.top_p > 1.f) {
    throw invalid_argument("Sampler: top_p must be in (0, 1]");
  }
  if (config_.top_k < 0) {
    throw invalid_argument("Sampler: top_k must be non-negative");
  }
}

// sample: int32_t — picks next token from logits [1, 1, vocab] or [1, vocab].
int32_t Sampler::sample(const Tensor& logits) const {
  vector<float> row = logits_vocab_row(logits);
  if (config_.greedy) {
    return greedy_pick(row);
  }
  apply_temperature(row, config_.temperature);
  apply_top_k(row, config_.top_k);
  vector<float> probs = softmax_probs(row);
  probs = apply_top_p(move(probs), config_.top_p);
  discrete_distribution<int> dist(probs.begin(), probs.end());
  return static_cast<int32_t>(dist(rng_));
}

}  // namespace serving
