#pragma once

#include "core/tensor.hpp"

#include <cstdint>
#include <vector>

using namespace std;

namespace serving {

// matmul: void — multiplies A[M,K] by B[K,N] and writes into out[M,N].
void matmul(const Tensor& a, const Tensor& b, Tensor& out);
// add_inplace: void — elementwise adds b into a (same numel).
void add_inplace(Tensor& a, const Tensor& b);
// mul_inplace: void — elementwise multiplies a by b (same numel).
void mul_inplace(Tensor& a, const Tensor& b);
// rmsnorm: void — applies RMSNorm with weight[dim] on x[batch,dim] into out.
void rmsnorm(const Tensor& x, const Tensor& weight, float eps, Tensor& out);
// softmax_last_dim: void — stable softmax along last dim of rank-2 x into out.
void softmax_last_dim(const Tensor& x, Tensor& out);
// silu_inplace: void — applies SiLU activation in place on x.
void silu_inplace(Tensor& x);
// embedding: void — gathers rows from weight[vocab,dim] for indices into out.
void embedding(const Tensor& weight, const vector<int32_t>& indices, Tensor& out);

}  // namespace serving
