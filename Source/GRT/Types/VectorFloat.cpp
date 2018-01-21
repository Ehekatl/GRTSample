#include "../GRT.h"
#include "VectorFloat.h"

namespace GRT {
VectorFloat::VectorFloat() {}

VectorFloat::VectorFloat(const size_type size) {
  resize(size);
}

VectorFloat::VectorFloat(const size_type size, const float& value) {
  resize(size, value);
}

VectorFloat::VectorFloat(const VectorFloat& rhs) : Vector(rhs) {}

VectorFloat::~VectorFloat() {
  clear();
}

VectorFloat& VectorFloat::operator=(const VectorFloat& rhs) {
  if (this != &rhs) {
    uint32 N = rhs.getSize();

    if (N > 0) {
      resize(N);
      std::copy(rhs.begin(), rhs.end(), this->begin());
    }
    else this->clear();
  }
  return *this;
}

VectorFloat& VectorFloat::operator=(const Vector<float>& rhs) {
  if (this != &rhs) {
    uint32 N = rhs.getSize();

    if (N > 0) {
      resize(N);
      std::copy(rhs.begin(), rhs.end(), this->begin());
    }
    else this->clear();
  }
  return *this;
}

bool VectorFloat::scale(const float minTarget,
                        const float maxTarget,
                        const bool  constrain) {
  MinMax range = getMinMax();

  return scale(range.minValue, range.maxValue, minTarget, maxTarget, constrain);
}

bool VectorFloat::scale(const float minSource,
                        const float maxSource,
                        const float minTarget,
                        const float maxTarget,
                        const bool  constrain) {
  const size_type N = this->size();

  if (N == 0) {
    return false;
  }

  size_type i    = 0;
  float    *data = getData();

  for (i = 0; i < N; i++) {
    data[i] = grt_scale(data[i],
                        minSource,
                        maxSource,
                        minTarget,
                        maxTarget,
                        constrain);
  }

  return true;
}

float VectorFloat::getMinValue() const {
  float minValue       = std::numeric_limits<float>::max();
  const size_type N    = this->size();
  const float    *data = getData();

  for (size_type i = 0; i < N; i++) {
    if (data[i] < minValue) minValue = data[i];
  }
  return minValue;
}

float VectorFloat::getMaxValue() const {
  float maxValue       = std::numeric_limits<float>::lowest();
  const size_type N    = this->size();
  const float    *data = getData();

  for (size_type i = 0; i < N; i++) {
    if (data[i] > maxValue) maxValue = data[i];
  }
  return maxValue;
}

float VectorFloat::getMean() const {
  float mean           = 0.0;
  const size_type N    = this->size();
  const float    *data = getData();

  for (size_type i = 0; i < N; i++) {
    mean += data[i];
  }
  mean /= N;

  return mean;
}

float VectorFloat::getStdDev() const {
  float mean           = getMean();
  float stdDev         = 0.0;
  const size_type N    = this->size();
  const float    *data = getData();

  for (size_type i = 0; i < N; i++) {
    stdDev += grt_sqr(data[i] - mean);
  }
  stdDev = grt_sqrt(stdDev / float(N - 1));

  return stdDev;
}

MinMax VectorFloat::getMinMax() const {
  const size_type N = this->size();
  MinMax range;

  if (N == 0) return range;

  const float *data = getData();

  for (size_type i = 0; i < N; i++) {
    range.updateMinMax(data[i]);
  }

  return range;
}
}
