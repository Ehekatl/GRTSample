#pragma once

#include "../GRT.h"

namespace GRT {
  // A common helper class to get min/max value
  class MinMax {
  public:
    MinMax() :minValue(0), maxValue(0) {};

    MinMax(float minValue, float maxValue) {
      this->minValue = minValue;
      this->maxValue = maxValue;
    }
    ~MinMax() {};

    MinMax& operator= (const MinMax &rhs) {
      if (this != &rhs) {
        this->minValue = rhs.minValue;
        this->maxValue = rhs.maxValue;
      }
      return *this;
    }

    bool updateMinMax(float newValue) {
      if (newValue < minValue) {
        minValue = newValue;
        return true;
      }
      if (newValue > maxValue) {
        maxValue = newValue;
        return true;
      }
      return false;
    }

    float minValue;
    float maxValue;
  };
}