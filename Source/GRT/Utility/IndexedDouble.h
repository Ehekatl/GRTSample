#pragma once

#include "../GRT.h"

namespace GRT {
class GRT_API IndexedDouble {
public:

  IndexedDouble() {
    index = 0;
    value = 0;
  }

  IndexedDouble(uint32 index, double value) {
    this->index = index;
    this->value = value;
  }

  IndexedDouble(const IndexedDouble& rhs) {
    this->index = rhs.index;
    this->value = rhs.value;
  }

  ~IndexedDouble() {}

  IndexedDouble& operator=(const IndexedDouble& rhs) {
    if (this != &rhs) {
      this->index = rhs.index;
      this->value = rhs.value;
    }
    return *this;
  }

  static bool sortIndexedDoubleByIndexDescending(IndexedDouble a,
                                                 IndexedDouble b) {
    return a.index > b.index;
  }

  static bool sortIndexedDoubleByIndexAscending(IndexedDouble a,
                                                IndexedDouble b) {
    return a.index < b.index;
  }

  static bool sortIndexedDoubleByValueDescending(IndexedDouble a,
                                                 IndexedDouble b) {
    return a.value > b.value;
  }

  static bool sortIndexedDoubleByValueAscending(IndexedDouble a,
                                                IndexedDouble b) {
    return a.value < b.value;
  }

  uint32 index;
  double value;
};
}
