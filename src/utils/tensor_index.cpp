#include "utils/tensor_index.hpp"

#include <stdexcept>

using namespace std;

namespace serving {

// linear_index: int64_t — maps multi-index to flat offset using shape and strides.
int64_t TensorIndex::linear_index(const vector<int64_t>& shape, const vector<int64_t>& strides,
                                  initializer_list<int64_t> indices) {
  if (indices.size() != shape.size()) {
    throw invalid_argument("rank mismatch");
  }
  int64_t idx = 0;
  size_t i = 0;
  for (int64_t d : indices) {
    if (d < 0 || d >= shape[i]) {
      throw out_of_range("index out of range");
    }
    idx += d * strides[i];
    ++i;
  }
  return idx;
}

}  // namespace serving
