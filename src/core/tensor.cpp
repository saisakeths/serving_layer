#include "core/tensor.hpp"
#include "utils/tensor_index.hpp"

#include <algorithm>
#include <stdexcept>

using namespace std;

namespace serving {

// Tensor: — constructs a tensor with the given shape and dtype.
Tensor::Tensor(vector<int64_t> shape, DType dtype) : shape_(move(shape)), dtype_(dtype) {
  if (shape_.empty()) {
    numel_ = 0;
    return;
  }
  numel_ = 1;
  for (int64_t d : shape_) {
    if (d < 0) {
      throw invalid_argument("negative dim");
    }
    numel_ *= static_cast<size_t>(d);
  }
  compute_strides();
  data_ = make_unique<float[]>(numel_);
}

// compute_strides: void — fills strides_ from shape_ for row-major layout.
void Tensor::compute_strides() {
  int size = shape_.size();
  strides_.assign(size, 1);
  if (shape_.size() <= 1) {
    return;
  }
  for (int i = size - 2; i >= 0; --i) {
    strides_[i] = strides_[i + 1] * shape_[i + 1];
  }
}

// zeros: Tensor — returns a tensor filled with 0.f.
Tensor Tensor::zeros(vector<int64_t> shape, DType dtype) {
  Tensor t(move(shape), dtype);
  for (size_t i = 0; i < t.numel_; ++i) {
    t.data_[i] = 0.f;
  }
  return t;
}

// from_vector: Tensor — builds a contiguous tensor from flat data matching shape numel.
Tensor Tensor::from_vector(vector<int64_t> shape, vector<float> data) {
  Tensor t(move(shape));
  if (data.size() != t.numel_) {
    throw invalid_argument("data size mismatch");
  }
  for (size_t i = 0; i < t.numel_; ++i) {
    t.data_[i] = data[i];
  }
  return t;
}

// at: float& — mutable element at multi-dimensional indices.
float& Tensor::at(initializer_list<int64_t> indices) {
  return data_[TensorIndex::linear_index(shape_, strides_, indices)];
}

// at: float — element at multi-dimensional indices.
float Tensor::at(initializer_list<int64_t> indices) const {
  return data_[TensorIndex::linear_index(shape_, strides_, indices)];
}

// view: Tensor — reshapes to new_shape when numel matches (copies for Phase 1).
Tensor Tensor::view(vector<int64_t> new_shape) const {
  int64_t new_numel = 1;
  for (int64_t d : new_shape) {
    new_numel *= d;
  }
  if (static_cast<size_t>(new_numel) != numel_) {
    throw invalid_argument("view size mismatch");
  }
  if (!contiguous_) {
    throw runtime_error("view requires contiguous tensor");
  }
  Tensor out;
  out.shape_ = move(new_shape);
  out.dtype_ = dtype_;
  out.numel_ = numel_;
  out.contiguous_ = true;
  out.compute_strides();
  out.data_ = make_unique<float[]>(numel_);
  copy(data_.get(), data_.get() + numel_, out.data_.get());
  return out;
}

}  // namespace serving
