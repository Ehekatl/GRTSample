#pragma once

#include "../GRT.h"
#include "../Types/VectorFloat.h"
#include "../Types/MatrixFloat.h"
#include "../Utility/Random.h"
#include <string>
#include <sstream>

namespace GRT {
class GRT_API Util {
public:

  /**
     Default constructor.
   */
  Util() {}

  /**
     Default destructor.
   */
  ~Util() {}

  /**
     Performs minmax scaling. The input value (x) will be scaled from the source
        range to the target range.

     @param x: the input value to be scaled
     @param minSource: the minimum source value (that x originates from)
     @param maxSource: the maximum source value (that x originates from)
     @param minTarget: the minimum target value (that x will be scaled to)
     @param maxTarget: the maximum target value (that x will be scaled to)
     @param constrain: if true, then the value will be constrained to the
        minSource and maxSource
     @return the scaled value
   */
  static float scale(const float& x,
                     const float& minSource,
                     const float& maxSource,
                     const float& minTarget,
                     const float& maxTarget,
                     const bool   constrain = false);

  /**
     Converts an int to a string.

     @param i: the value you want to convert to a string
     @return FString: the value as a string
   */
  static FString intToString(const int& i);

  /**
     Converts an unsigned int to a string.

     @param i: the value you want to convert to a string
     @return FString: the value as a string
   */
  static FString intToString(const uint32& i);

  /**
     Converts an unsigned int to a string.

     @param i: the value you want to convert to a string
     @return FString: the value as a string
   */
  static FString toString(const int& i);

  /**
     Converts an unsigned int to a string.

     @param i: the value you want to convert to a string
     @return FString: the value as a string
   */
  static FString toString(const uint32& i);

  /**
     Converts a boolean to a string.

     @param b: the value you want to convert to a string
     @return FString: the boolan as a string
   */
  static FString toString(const bool& b);

  /**
     Converts a float to a string.

     @param v: the value you want to convert to a string
     @return FString: the value as a string
   */
  static FString toString(const float& v);

  /**
     Converts a double to a string.

     @param v: the value you want to convert to a string
     @return FString: the value as a string
   */
  static FString toString(const double& v);

  /**
     Converts a string to an int.

     @param s: the value you want to convert to an int
     @return int: the value as an int
   */
  static int     stringToInt(const FString& s);

  /**
     Converts a string to a double.

     @param s: the value you want to convert to a double
     @return the value as a double
   */
  static double  stringToDouble(const FString& s);

  /**
     Converts a string to a float.

     @param s: the value you want to convert to a float
     @return the value as a float
   */
  static float   stringToFloat(const FString& s);

  /**
     Converts a string to a boolean. Any string that matches true, True, TRUE,
        t, T, or 1 will return true, anything else will return false.

     @param s: the value you want to convert to a bool
     @return bool: the value as a bool
   */
  static bool    stringToBool(const FString& s);


  /**
     Limits the input value so it is between the range of minValue and maxValue.
     If the input value is below the minValue then the output of the function
        will be the minValue.
     If the input value is above the maxValue then the output of the function
        will be the maxValue.
     Otherwise, the out of the function will be the input.

     @param value: the input value that should be limited
     @param minValue: the minimum value that should be limited
     @param maxValue: the maximum value that should be limited
     @return the limited double input value
   */
  static float limit(const float value,
                     const float minValue,
                     const float maxValue);

  /**
     Computes the sum of the vector x.

     @param x: the vector of values you want to sum
     @return double: the sum of the input vector x
   */
  static float sum(const VectorFloat& x);

  /**
     Computes the dot product between the two input vectors. The two input
        vectors must have the same size.

     @param a: the first vector for the dot product
     @param b: the second vector for the dot product
     @return double: the dot product between the two input vectors, if the two
        input vectors are not the same size then the dist will be INF
   */
  static float dotProduct(const VectorFloat& a,
                          const VectorFloat& b);

  /**
     Computes the euclidean distance between the two input vectors. The two
        input vectors must have the same size.

     @param a: the first vector for the euclidean distance
     @param b: the second vector for the euclidean distance
     @return the euclidean distance between the two input vectors, if the two
        input vectors are not the same size then the dist will be INF
   */
  static float euclideanDistance(const VectorFloat& a,
                                 const VectorFloat& b);

  /**
     Computes the squared euclidean distance between the two input vectors. The
        two input vectors must have the same size.

     @param a: the first vector for the euclidean distance
     @param b: the second vector for the euclidean distance
     @return the euclidean distance between the two input vectors, if the two
        input vectors are not the same size then the dist will be INF
   */
  static float squaredEuclideanDistance(const VectorFloat& a,
                                        const VectorFloat& b);

  /**
     Computes the manhattan distance between the two input vectors. The two
        input vectors must have the same size.
     The manhattan distance is also known as the L1 norm, taxicab distance, city
        block distance, or rectilinear distance.

     @param a: the first vector for the manhattan distance
     @param b: the second vector for the manhattan distance
     @return the manhattan distance between the two input vectors, if the two
        input vectors are not the same size then the dist will be INF
   */
  static float manhattanDistance(const VectorFloat& a,
                                 const VectorFloat& b);

  /**
     Computes the cosine distance between the two input vectors. The two input
        vectors must have the same size.
     The cosine distance can be used as a similarity measure, the distance
        ranges from −1 meaning exactly opposite, to 1 meaning exactly the same,
     with 0 usually indicating independence, and in-between values indicating
        intermediate similarity or dissimilarity.

     @param a: the first vector for the cosine distance
     @param b: the second vector for the cosine distance
     @return the cosine distance between the two input vectors, if the two input
        vectors are not the same size then the dist will be INF
   */
  static float cosineDistance(const VectorFloat& a,
                              const VectorFloat& b);

  /**
     Scales the vector from a source range to the new target range

     @param x: the input value to be scaled
     @param minSource: the minimum source value (that x originates from)
     @param maxSource: the maximum source value (that x originates from)
     @param minTarget: the minimum target value (that x will be scaled to)
     @param maxTarget: the maximum target value (that x will be scaled to)
     @param constrain: if true, then the value will be constrained to the
        minSource and maxSource
     @return the scaled input vector
   */
  static VectorFloat scale(const VectorFloat& x,
                           const float        minSource,
                           const float        maxSource,
                           const float        minTarget = 0,
                           const float        maxTarget = 1,
                           const bool         constrain = false);

  /**
     Normalizes the input vector x so the sum is 1.

     @param x: the vector of values you want to normalize
     @return the normalized input vector (the sum of which should be 1)
   */
  static VectorFloat normalize(const VectorFloat& x);

  /**
     Limits the input data x so each element is within the range [minValue
        maxValue].
     Returns a new vector with the limited data.

     @param x: the vector of values you want to limit
     @param minValue: the minimum value
     @param maxValue: the maximum value
     @return the limited input vector
   */
  static VectorFloat limit(const VectorFloat& x,
                           const float        minValue,
                           const float        maxValue);

  /**
     Gets the minimum value in the input vector.

     @param x: the vector of values you want to find the minimum value for
     @return the minimum value in the input vector, this will be INF if the
        input vector size is 0
   */
  static float  getMin(const VectorFloat& x);

  /**
     Gets the index of the minimum value in the input vector.

     @param x: the vector of values you want to find the minimum index value for
     @return the index of the minimum value in the vector
   */
  static uint32 getMinIndex(const VectorFloat& x);

  /**
     Gets the maximum value in the input vector.

     @param x: the vector of values you want to find the maximum value for
     @return the maximum value in the input vector, this will be INF if the
        input vector size is 0
   */
  static float  getMax(const VectorFloat& x);

  /**
     Gets the index of the maximum value in the input vector.

     @param x: the vector of values you want to find the maximum index value for
     @return the index of the maximum value in the vector
   */
  static uint32 getMaxIndex(const VectorFloat& x);

  /**
     Gets the minimum value in the input vector.

     @param x: the vector of values you want to find the minimum value for
     @return the minimum value in the input vector, this will be INF if the
        input vector size is 0
   */
  static uint32 getMin(const std::vector<unsigned int>& x);

  /**
     Gets the maximum value in the input vector.

     @param x: the vector of values you want to find the maximum value for
     @return the maximum value in the input vector, this will be INF if the
        input vector size is 0
   */
  static uint32 getMax(const std::vector<unsigned int>& x);

  /**
     Converts the cartesian values {x y} into polar values {r theta}

     @param x: the x cartesian value
     @param y: the y cartesian value
     @param r: the return radius value
     @param theta: the return theta value
     @return void
   */
  static void   cartToPolar(const float x,
                            const float y,
                            float     & r,
                            float     & theta);

  /**
     Converts the polar values {r theta} into the cartesian values {x y}.

     @param r: the radius polar value
     @param theta: the theta polar value
     @param x: the return x value
     @param y: the return y value
     @return void
   */
  static void polarToCart(const float r,
                          const float theta,
                          float     & x,
                          float     & y);
};
}
