#pragma once

#include <cstdint>

namespace serving {

enum class DType { Float32, Float16 };

// dtype_size: size_t — returns storage size in bytes for the given dtype.
size_t dtype_size(DType dt);

}  // namespace serving
