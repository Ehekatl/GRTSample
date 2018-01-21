#pragma once

#include "../GRT.h"
#include <cmath>
#include <limits>

namespace GRT {
  #ifndef PI
  # define PI 3.14159265358979323846264338327950288
  #endif // ifndef PI

  #ifndef TWO_PI
  # define TWO_PI 6.28318530718
  #endif // ifndef TWO_PI

  #ifndef ONE_OVER_TWO_PI
  # define ONE_OVER_TWO_PI (1.0 / TWO_PI)
  #endif // ifndef ONE_OVER_TWO_PI

  #ifndef SQRT_TWO_PI
  # define SQRT_TWO_PI 2.506628274631
  #endif // ifndef SQRT_TWO_PI

  #ifndef grt_min
  # define grt_min(a, b) (((a) < (b)) ? (a) : (b))
  #endif // ifndef grt_min

  #ifndef grt_max
  # define grt_max(a, b) (((a) > (b)) ? (a) : (b))
  #endif // ifndef grt_max


  #ifndef NAN
static const unsigned long __nan[2] = { 0xffffffff, 0x7fffffff };
  # define NAN (*(const float *)__nan)
  #endif // NAN

  #ifndef grt_isnan
  # define grt_isnan(x) (x != x)
  #endif // grt_isnan

  #ifndef grt_isinf
  # define grt_isinf(x) (!grt_isnan(x) && grt_isnan(x - x))
  #endif // grt_isinf

  #ifndef INFINITY
  # define INFINITY (DBL_MAX + DBL_MAX)
  #endif // INFINITY

template<class T>inline T SQR(const T& a) {
  return a * a;
}

template<class T>inline void SWAP(T& a, T& b) {
  T temp(a); a = b; b = temp;
}

template<class T>inline T SIGN(const T& a, const T& b) {
  return b >= 0 ? (a >= 0 ? a : -a) : (a >= 0 ? -a : a);
}

template<class T>inline void grt_swap(T& a, T& b) {
  T temp(a); a = b; b = temp;
}

template<class T>class grt_numeric_limits {
public:

  static T min() {
    return std::numeric_limits<T>::min();
  }

  static T max() {
    return std::numeric_limits<T>::max();
  }
};

inline float grt_sqr(const float& x) {
  return x * x;
}

inline float grt_sqrt(const float& x) {
  return sqrt(x);
}

inline float grt_antilog(const float& x) {
  return exp(x);
}

inline float grt_exp(const float& x) {
  return exp(x);
}

inline float grt_log(const float& x) {
  return log(x);
}

inline float grt_sigmoid(const float& x) {
  return 1.0 / (1.0 + exp(-x));
}

template<class T>
T grt_scale(const T  & x,
            const T  & minSource,
            const T  & maxSource,
            const T  & minTarget,
            const T  & maxTarget,
            const bool constrain = false) {
  if (constrain) {
    if (x <= minSource) return minTarget;

    if (x >= maxSource) return maxTarget;
  }

  if (minSource == maxSource) return minTarget;

  return (((x - minSource) * (maxTarget - minTarget)) / (maxSource - minSource)) +
         minTarget;
}
}
