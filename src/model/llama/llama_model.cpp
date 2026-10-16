#include "model/llama/llama_model.hpp"

#include "ops/ops.hpp"

#include <algorithm>
#include <stdexcept>

using namespace std;

namespace serving {

// LlamaModel: — stacks transformer blocks and output head from config and weights.
LlamaModel::LlamaModel(LlamaConfig config, LlamaWeights weights)
    : config_(config), weights_(move(weights)) {
  config_.validate();
  validate_llama_weights(config_, weights_);
  BlockConfig bc = config_.block_config();
  blocks_.reserve(static_cast<size_t>(config_.n_layer));
  for (int64_t i = 0; i < config_.n_layer; ++i) {
    blocks_.emplace_back(bc, move(weights_.layers[static_cast<size_t>(i)]));
  }
}

// head_weight: const Tensor& — lm_head or tied tok_embeddings.
const Tensor& LlamaModel::head_weight() const {
  if (config_.tie_word_embeddings) {
    return weights_.tok_embeddings;
  }
  return weights_.lm_head;
}

// forward: void — prefill with KV cache; logits shape [1, seq, vocab_size].
void LlamaModel::forward(const vector<int32_t>& input_ids, const vector<int64_t>& positions,
                         LlamaCache& cache, Tensor& logits) const {
  const int64_t seq = static_cast<int64_t>(input_ids.size());
  if (static_cast<int64_t>(positions.size()) != seq) {
    throw invalid_argument("forward: positions length");
  }
  if (seq > cache.max_seq()) {
    throw invalid_argument("forward: seq exceeds cache max_seq");
  }
  const int64_t batch = 1;
  const int64_t dim = config_.dim;
  const int64_t vocab = config_.vocab_size;

  if (logits.shape() != vector<int64_t>{batch, seq, vocab}) {
    throw invalid_argument("forward: logits shape");
  }

  Tensor embed_2d({seq, dim});
  embedding(weights_.tok_embeddings, input_ids, embed_2d);

  Tensor hidden_a({batch, seq, dim});
  Tensor hidden_b({batch, seq, dim});
  for (int64_t s = 0; s < seq; ++s) {
    for (int64_t d = 0; d < dim; ++d) {
      hidden_a.at({0, s, d}) = embed_2d.at({s, d});
    }
  }

  Tensor* in = &hidden_a;
  Tensor* out = &hidden_b;
  for (size_t i = 0; i < blocks_.size(); ++i) {
    blocks_[i].forward_prefill(*in, positions, cache.layer(i), *out);
    swap(in, out);
  }
  const Tensor& final_hidden = *in;

  Tensor flat = final_hidden.reshape({batch * seq, dim});
  Tensor normed({batch * seq, dim});
  rmsnorm(flat, weights_.output_norm, config_.rms_eps, normed);

  Tensor logits_2d({batch * seq, vocab});
  linear(normed, head_weight(), logits_2d);

  for (int64_t s = 0; s < seq; ++s) {
    for (int64_t v = 0; v < vocab; ++v) {
      logits.at({0, s, v}) = logits_2d.at({s, v});
    }
  }
}

}  // namespace serving
