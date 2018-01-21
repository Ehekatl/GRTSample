#pragma once

#include "../GRT.h"
#include "../Types/VectorFloat.h"
#include "../Types/MatrixFloat.h"

namespace GRT {
enum DataType { DATA_TYPE_UNKNOWN = 0, DATA_TYPE_VECTOR, DATA_TYPE_MATRIX };

// TODO
// FIXME: Replace the dynamic_cast ?

template<typename T>
const T data_type_cast(const void *data) {
  return dynamic_cast<T>(data);
}

template<typename T>
T data_type_cast(void *data) {
  return dynamic_cast<T>(data);
}
}
