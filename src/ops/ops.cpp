#include "ops/ops.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

using namespace std;

namespace serving {

// matmul: void — multiplies A[M,K] by B[K,N] and writes into out[M,N].
void matmul(const Tensor& a, const Tensor& b, Tensor& out) {
  const auto& as = a.shape();
  const auto& bs = b.shape();
  if (as.size() != 2 || bs.size() != 2) {
    throw invalid_argument("matmul expects rank-2");
  }
  int64_t M = as[0], K = as[1], K2 = bs[0], N = bs[1];
  if (K != K2) {
    throw invalid_argument("inner dim mismatch");
  }
  if (out.shape() != vector<int64_t>{M, N}) {
    throw invalid_argument("out shape mismatch");
  }
  const float* ap = a.data();
  const float* bp = b.data();
  float* op = out.data();
  fill(op, op + M * N, 0.f);
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t k = 0; k < K; ++k) {
      float av = ap[i * K + k];
      for (int64_t j = 0; j < N; ++j) {
        op[i * N + j] += av * bp[k * N + j];
      }
    }
  }
}

// add_inplace: void — elementwise adds b into a (same numel).
void add_inplace(Tensor& a, const Tensor& b) {
  if (a.numel() != b.numel()) {
    throw invalid_argument("add size mismatch");
  }
  for (size_t i = 0; i < a.numel(); ++i) {
    a.data()[i] += b.data()[i];
  }
}

// mul_inplace: void — elementwise multiplies a by b (same numel).
void mul_inplace(Tensor& a, const Tensor& b) {
  if (a.numel() != b.numel()) {
    throw invalid_argument("mul size mismatch");
  }
  for (size_t i = 0; i < a.numel(); ++i) {
    a.data()[i] *= b.data()[i];
  }
}

// rmsnorm: void — applies RMSNorm with weight[dim] on x[batch,dim] into out.
void rmsnorm(const Tensor& x, const Tensor& weight, float eps, Tensor& out) {
  const auto& xs = x.shape();
  if (xs.size() != 2) {
    throw invalid_argument("rmsnorm expects [batch, dim]");
  }
  int64_t batch = xs[0], dim = xs[1];
  if (weight.shape() != vector<int64_t>{dim}) {
    throw invalid_argument("weight shape");
  }
  if (out.shape() != xs) {
    throw invalid_argument("out shape");
  }
  for (int64_t b = 0; b < batch; ++b) {
    float sum_sq = 0.f;
    for (int64_t d = 0; d < dim; ++d) {
      float v = x.at({b, d});
      sum_sq += v * v;
    }
    float scale = 1.f / sqrt(sum_sq / static_cast<float>(dim) + eps);
    for (int64_t d = 0; d < dim; ++d) {
      out.at({b, d}) = x.at({b, d}) * scale * weight.at({d});
    }
  }
}

// softmax_last_dim: void — stable softmax along last dim of rank-2 x into out.
void softmax_last_dim(const Tensor& x, Tensor& out) {
  const auto& xs = x.shape();
  if (xs.size() != 2) {
    throw invalid_argument("softmax expects rank 2");
  }
  int64_t rows = xs[0], cols = xs[1];
  if (out.shape() != xs) {
    throw invalid_argument("out shape");
  }
  for (int64_t r = 0; r < rows; ++r) {
    float max_v = x.at({r, 0});
    for (int64_t c = 1; c < cols; ++c) {
      max_v = max(max_v, x.at({r, c}));
    }
    float sum = 0.f;
    for (int64_t c = 0; c < cols; ++c) {
      float e = exp(x.at({r, c}) - max_v);
      out.at({r, c}) = e;
      sum += e;
    }
    for (int64_t c = 0; c < cols; ++c) {
      out.at({r, c}) /= sum;
    }
  }
}

// silu_inplace: void — applies SiLU activation in place on x.
void silu_inplace(Tensor& x) {
  for (size_t i = 0; i < x.numel(); ++i) {
    float v = x.data()[i];
    x.data()[i] = v / (1.f + exp(-v));
  }
}

// embedding: void — gathers rows from weight[vocab,dim] for indices into out.
void embedding(const Tensor& weight, const vector<int32_t>& indices, Tensor& out) {
  int64_t vocab = weight.shape()[0];
  int64_t dim = weight.shape()[1];
  if (out.shape() != vector<int64_t>{static_cast<int64_t>(indices.size()), dim}) {
    throw invalid_argument("embedding out shape");
  }
  for (size_t i = 0; i < indices.size(); ++i) {
    int32_t id = indices[i];
    if (id < 0 || id >= vocab) {
      throw out_of_range("token id");
    }
    for (int64_t d = 0; d < dim; ++d) {
      out.at({static_cast<int64_t>(i), d}) = weight.at({id, d});
    }
  }
}

// linear: void — computes out = x @ weight^T for x[M,K], weight[N,K], out[M,N].
void linear(const Tensor& x, const Tensor& weight, Tensor& out) {
  const auto& xs = x.shape();
  const auto& ws = weight.shape();
  if (xs.size() != 2 || ws.size() != 2) {
    throw invalid_argument("linear expects rank-2");
  }
  int64_t M = xs[0], K = xs[1], N = ws[0], K2 = ws[1];
  if (K != K2) {
    throw invalid_argument("linear inner dim mismatch");
  }
  if (out.shape() != vector<int64_t>{M, N}) {
    throw invalid_argument("linear out shape");
  }
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      float sum = 0.f;
      for (int64_t k = 0; k < K; ++k) {
        sum += x.at({m, k}) * weight.at({n, k});
      }
      out.at({m, n}) = sum;
    }
  }
}

}  // namespace serving
