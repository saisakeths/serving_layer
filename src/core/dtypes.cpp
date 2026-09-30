#include "core/dtypes.hpp"

using namespace std;

namespace serving {

// dtype_size: size_t — returns storage size in bytes for the given dtype.
size_t dtype_size(DType dt) {
  switch (dt) {
    case DType::Float32:
      return 4;
    case DType::Float16:
      return 2;
  }
  return 4;
}

}  // namespace serving
