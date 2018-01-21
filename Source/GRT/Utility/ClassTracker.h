#pragma once

#include "../GRT.h"

namespace GRT {
class ClassTracker {
public:

  ClassTracker(uint32  classLabel = 0,
               uint32  counter = 0,
               FString className = "NOT_SET") {
    this->classLabel = classLabel;
    this->counter    = counter;
    this->className  = className;
  }

  ClassTracker(const ClassTracker& rhs) {
    this->classLabel = rhs.classLabel;
    this->counter    = rhs.counter;
    this->className  = rhs.className;
  }

  ~ClassTracker() {}

  ClassTracker& operator=(const ClassTracker& rhs) {
    if (this != &rhs) {
      this->classLabel = rhs.classLabel;
      this->counter    = rhs.counter;
      this->className  = rhs.className;
    }
    return *this;
  }

  static bool sortByClassLabelDescending(ClassTracker a, ClassTracker b) {
    return a.classLabel > b.classLabel;
  }

  static bool sortByClassLabelAscending(ClassTracker a, ClassTracker b) {
    return a.classLabel < b.classLabel;
  }

  uint32  classLabel;
  uint32  counter;
  FString className;
};
}
