#pragma once

#include "core/dtypes.hpp"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <vector>
using namespace std;

namespace serving {

class Tensor {
 public:
  Tensor() = default;
  // Tensor: — constructs a tensor with the given shape and dtype (data uninitialized).
  Tensor(vector<int64_t> shape, DType dtype = DType::Float32);

  // zeros: Tensor — returns a tensor filled with 0.f.
  static Tensor zeros(vector<int64_t> shape, DType dtype = DType::Float32);
  // from_vector: Tensor — builds a contiguous tensor from flat data matching shape numel.
  static Tensor from_vector(vector<int64_t> shape, vector<float> data);

  // shape: const vector<int64_t>& — returns tensor dimensions.
  const vector<int64_t>& shape() const { return shape_; }
  // strides: const vector<int64_t>& — returns row-major stride per dimension.
  const vector<int64_t>& strides() const { return strides_; }
  // dtype: DType — returns element type.
  DType dtype() const { return dtype_; }
  // numel: size_t — returns total element count.
  size_t numel() const { return numel_; }
  // is_contiguous: bool — true if layout is dense row-major.
  bool is_contiguous() const { return contiguous_; }

  // data: float* — mutable pointer to storage.
  float* data() { return data_.get(); }
  // data: const float* — read-only pointer to storage.
  const float* data() const { return data_.get(); }

  // at: float& — mutable element at multi-dimensional indices.
  float& at(initializer_list<int64_t> indices);
  // at: float — element at multi-dimensional indices.
  float at(initializer_list<int64_t> indices) const;

  // view: Tensor — reshapes to new_shape when numel matches (copies for Phase 1).
  Tensor view(vector<int64_t> new_shape) const;

 private:
  // compute_strides: void — fills strides_ from shape_ for row-major layout.
  void compute_strides();

  vector<int64_t> shape_;
  vector<int64_t> strides_;
  DType dtype_ = DType::Float32;
  size_t numel_ = 0;
  bool contiguous_ = true;
  unique_ptr<float[]> data_;
};

}  // namespace serving
