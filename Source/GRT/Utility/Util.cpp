#include "../GRT.h"
#include "Util.h"

namespace GRT {
float Util::scale(const float& x,
                  const float& minSource,
                  const float& maxSource,
                  const float& minTarget,
                  const float& maxTarget,
                  const bool   constrain) {
  if (constrain) {
    if (x <= minSource) return minTarget;

    if (x >= maxSource) return maxTarget;
  }

  if (minSource == maxSource) return minTarget;

  return (((x - minSource) * (maxTarget - minTarget)) / (maxSource - minSource)) +
         minTarget;
}

FString Util::intToString(const int& i) {
  std::stringstream s;

  s << i;
  return FString(s.str().c_str());
}

FString Util::intToString(const uint32& i) {
  std::stringstream s;

  s << i;
  return FString(s.str().c_str());
}

FString Util::toString(const int& i) {
  std::stringstream s;

  s << i;
  return FString(s.str().c_str());
}

FString Util::toString(const uint32& i) {
  std::stringstream s;

  s << i;
  return FString(s.str().c_str());
}

FString Util::toString(const bool& b) {
  return b ? TEXT("True") : TEXT("False");
}

FString Util::toString(const float& v) {
  std::stringstream s;

  s << v;
  return FString(s.str().c_str());
}

int Util::stringToInt(const FString& value) {
  int i = FCString::Atoi(*value);

  return i;
}

float Util::stringToFloat(const FString& value) {
  float f = FCString::Atof(*value);

  return f;
}

double Util::stringToDouble(const FString& value) {
  double d = FCString::Atod(*value);

  return d;
}

bool Util::stringToBool(const FString& value) {
  if (value == "true") return true;

  if (value == "True") return true;

  if (value == "TRUE") return true;

  if (value == "t") return true;

  if (value == "T") return true;

  if (value == "1") return true;

  return false;
}

float Util::limit(const float value, const float minValue, const float maxValue) {
  if (value <= minValue) return minValue;

  if (value >= maxValue) return maxValue;

  return value;
}

float Util::sum(const VectorFloat& x) {
  float s       = 0;
  std::size_t N = x.size();

  for (std::size_t i = 0; i < N; i++) s += x[i];
  return s;
}

float Util::dotProduct(const VectorFloat& a, const VectorFloat& b) {
  if (a.size() != b.size()) return std::numeric_limits<float>::max();

  std::size_t N = a.size();
  float d       = 0;

  for (std::size_t i = 0; i < N; i++) {
    d += a[i] * b[i];
  }
  return d;
}

float Util::euclideanDistance(const VectorFloat& a, const VectorFloat& b) {
  if (a.size() != b.size()) return std::numeric_limits<float>::max();

  std::size_t N = a.size();
  float d       = 0;

  for (std::size_t i = 0; i < N; i++) {
    d += (a[i] - b[i]) * (a[i] - b[i]);
  }
  return sqrt(d);
}

float Util::squaredEuclideanDistance(const VectorFloat& a, const VectorFloat& b) {
  if (a.size() != b.size()) return std::numeric_limits<float>::max();

  std::size_t N = a.size();
  float d       = 0;

  for (std::size_t i = 0; i < N; i++) {
    d += (a[i] - b[i]) * (a[i] - b[i]);
  }
  return d;
}

float Util::manhattanDistance(const VectorFloat& a, const VectorFloat& b) {
  if (a.size() != b.size()) return std::numeric_limits<float>::max();

  std::size_t N = a.size();
  float d       = 0;

  for (std::size_t i = 0; i < N; i++) {
    d += fabs(a[i] - b[i]);
  }
  return d;
}

float Util::cosineDistance(const VectorFloat& a, const VectorFloat& b) {
  if (a.size() != b.size()) return std::numeric_limits<float>::max();

  std::size_t N    = a.size();
  float dotProduct = 0;
  float aSum       = 0;
  float bSum       = 0;

  for (std::size_t i = 0; i < N; i++) {
    dotProduct += a[i] * b[i];
    aSum       += a[i] * a[i];
    bSum       += b[i] * b[i];
  }
  return dotProduct / sqrt(aSum * bSum);
}

VectorFloat Util::scale(const VectorFloat& x,
                        const float        minSource,
                        const float        maxSource,
                        const float        minTarget,
                        const float        maxTarget,
                        const bool         constrain) {
  std::size_t N = x.size();
  VectorFloat y(N);

  for (std::size_t i = 0; i < N; i++) {
    y[i] = scale(x[i], minSource, maxSource, minTarget, maxTarget, constrain);
  }
  return y;
}

VectorFloat Util::normalize(const VectorFloat& x) {
  std::size_t N = x.size();
  VectorFloat y(N);
  float s = 0;

  for (std::size_t i = 0; i < N; i++) s += x[i];

  if (s != 0) {
    for (std::size_t i = 0; i < N; i++) y[i] = x[i] / s;
  }
  else {
    for (std::size_t i = 0; i < N; i++) y[i] = 0;
  }
  return y;
}

VectorFloat Util::limit(const VectorFloat& x,
                        const float        minValue,
                        const float        maxValue) {
  std::size_t N = x.size();
  VectorFloat y(N);

  for (std::size_t i = 0; i < N; i++) y[i] = limit(x[i], minValue, maxValue);
  return y;
}

float Util::getMin(const VectorFloat& x) {
  float min     = std::numeric_limits<float>::max();
  std::size_t N = x.size();

  for (std::size_t i = 0; i < N; i++) {
    if (x[i] < min) {
      min = x[i];
    }
  }
  return min;
}

unsigned int getMinIndex(const VectorFloat& x) {
  unsigned int minIndex = 0;
  float min             = std::numeric_limits<float>::max();
  unsigned int N        = (unsigned int)x.size();

  for (unsigned int i = 0; i < N; i++) {
    if (x[i] < min) {
      min      = x[i];
      minIndex = i;
    }
  }
  return minIndex;
}

float Util::getMax(const VectorFloat& x) {
  float max     = std::numeric_limits<float>::min();
  std::size_t N = x.size();

  for (std::size_t i = 0; i < N; i++) {
    if (x[i] > max) {
      max = x[i];
    }
  }
  return max;
}

unsigned int Util::getMaxIndex(const VectorFloat& x) {
  unsigned int maxIndex = 0;
  float max             = std::numeric_limits<float>::min();
  unsigned int N        = (unsigned int)x.size();

  for (unsigned int i = 0; i < N; i++) {
    if (x[i] > max) {
      max      = x[i];
      maxIndex = i;
    }
  }
  return maxIndex;
}

unsigned int Util::getMin(const std::vector<unsigned int>& x) {
  unsigned int min    = std::numeric_limits<unsigned int>::max();
  const std::size_t N = x.size();

  for (std::size_t i = 0; i < N; i++) {
    if (x[i] < min) {
      min = x[i];
    }
  }
  return min;
}

unsigned int Util::getMax(const std::vector<unsigned int>& x) {
  unsigned int max    = std::numeric_limits<unsigned int>::min();
  const std::size_t N = x.size();

  for (size_t i = 0; i < N; i++) {
    if (x[i] > max) {
      max = x[i];
    }
  }
  return max;
}

void Util::cartToPolar(const float x, const float y, float& r, float& theta) {
#ifndef PI
  float PI = 3.14159265358979323846;
#endif // ifndef PI

#ifndef TWO_PI
  float TWO_PI = 6.28318530718;
#endif // ifndef TWO_PI

  r     = 0;
  theta = 0;

  // Compute r
  r = sqrt((x * x) + (y * y));

  // Compute theta
  int type = 0;

  if ((x > 0) && (y >= 0)) type = 1;

  if ((x > 0) && (y < 0)) type = 2;

  if (x < 0) type = 3;

  if ((x == 0) && (y > 0)) type = 4;

  if ((x == 0) && (y < 0)) type = 5;

  if ((x == 0) && (y == 0)) type = 6;

  switch (type) {
  case (1):
    theta = atan(y / x) * (180.0 / PI);
    break;

  case (2):
    theta = (atan(y / x) + TWO_PI) * (180.0 / PI);
    break;

  case (3):
    theta = (atan(y / x) + PI) * (180.0 / PI);
    break;

  case (4):
    theta = (PI / 2.0) * (180.0 / PI);
    break;

  case (5):
    theta = ((3 * PI) / 2.0) * (180.0 / PI);
    break;

  case (6):
    theta = 0.0;
    break;

  default:
    theta = 0.0;
    break;
  }
}

void Util::polarToCart(const float r, const float theta, float& x, float& y) {
  x = r * cos(theta);
  y = r * sin(theta);
}
}
