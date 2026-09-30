#pragma once

#include <cstdint>
#include <initializer_list>
#include <vector>

using namespace std;

namespace serving {

// TensorIndex — helpers for mapping multi-dimensional indices to flat tensor offsets.
class TensorIndex {
 public:
  // linear_index: int64_t — maps multi-index to flat offset using shape and strides.
  static int64_t linear_index(const vector<int64_t>& shape, const vector<int64_t>& strides,
                              initializer_list<int64_t> indices);
};

}  // namespace serving
