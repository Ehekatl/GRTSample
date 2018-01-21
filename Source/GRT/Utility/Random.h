#pragma once

#include "../GRT.h"
#include "../Types/Vector.h"
#include "../Types/VectorFloat.h"
#include "IndexedDouble.h"
#include <random>

namespace GRT {
class GRT_API Random {
public:

  Random();

  /**
     Default destructor.
   */
  ~Random();

  /**
     Gets a random integer in the range [minRange maxRange-1], using a uniform
        distribution

     @param minRange: the minimum value in the range (inclusive)
     @param maxRange: the maximum value in the range (not inclusive)
     @return returns an integer in the range [minRange maxRange-1]
   */
  int getRandomNumberInt(int minRange,
                         int maxRange);

  /**
     Gets a random integer from the Vector values. The probability of choosing a
        specific integer from the
     values Vector is given by the corresponding weight in the weights Vector.
        The size of the values
     Vector must match the size of the weights Vector. The weights do not need
        to sum to 1.

     For example, if the input values are: [1 2 3] and weights are: [0.7 0.2
        0.1], then the 1 value would
     be randomly returned 70% of the time, the 2 value returned 20% of the time
        and the 3 value returned
     10% of the time.

     @param values: a Vector containing the N possible values the function can
        return
     @param weights: the corresponding weights for the values Vector (must be
        the same size as the values Vector)
     @return returns a random integer from the values Vector, with a probability
        relative to the values weight
   */
  int getRandomNumberWeighted(const Vector<int>& values,
                              const VectorFloat& weights);

  /**
     Gets a random integer from the input Vector. The probability of choosing a
        specific integer is given by the
     corresponding weight of that value. The weights do not need to sum to 1.

     For example, if the input values are: [{1 0.7},{2 0.2}, {3 0.1}], then the
        1 value would be randomly returned
     70% of the time, the 2 value returned 20% of the time and the 3 value
        returned 10% of the time.

     @param weightedValues: a Vector of IndexedDouble values, the (int) indexs
        represent the value that will be returned while the (float) values
        represent the weight of choosing that specific index
     @return returns a random integer from the values Vector, with a probability
        relative to the values weight
   */
  int getRandomNumberWeighted(Vector<IndexedDouble>weightedValues);

  /**
     This function is similar to the getRandomNumberWeighted(Vector<
        IndexedDouble > weightedValues), with the exception that the user needs
     to sort the weightedValues Vector and create the accumulated lookup table
        (x). This is useful if you need to call the same function
     multiple times on the same weightedValues, allowing you to only sort and
        build the loopup table once.

     Gets a random integer from the input Vector. The probability of choosing a
        specific integer is given by the
     corresponding weight of that value. The weights do not need to sum to 1.

     For example, if the input values are: [{1 0.7},{2 0.2}, {3 0.1}], then the
        1 value would be randomly returned
     70% of the time, the 2 value returned 20% of the time and the 3 value
        returned 10% of the time.

     @param weightedValues: a sorted Vector of IndexedDouble values, the (int)
        indexs represent the value that will be returned while the (float)
        values represent the weight of choosing that specific index
     @param x: a Vector containing the accumulated lookup table
     @return returns a random integer from the values Vector, with a probability
        relative to the values weight
   */
  int getRandomNumberWeighted(Vector<IndexedDouble>& weightedValues,
                              VectorFloat          & x);

  /**
     Gets a random float in the range [minRange maxRange], using a uniform
        distribution

     @param minRange: the minimum value in the range (inclusive)
     @param maxRange: the maximum value in the range (inclusive)
     @return returns a float in the range [minRange maxRange]
   */
  float getRandomNumberUniform(float minRange = 0.0,
                               float maxRange = 1.0);

  /**
     Gets a random float in the range [minRange maxRange], using a uniform
        distribution

     @param minRange: the minimum value in the range (inclusive)
     @param maxRange: the maximum value in the range (inclusive)
     @return returns a float in the range [minRange maxRange]
   */
  float getUniform(const float& minRange = 0.0,
                   const float& maxRange = 1.0);

  /**
     Gets a random float, using a Gaussian distribution with mu 0 and sigma 1.0

     @param mu: the mu parameter for the Gaussian distribution
     @param sigma: the sigma parameter for the Gaussian distribution
     @return returns a float from the Gaussian distribution controlled by mu and
        sigma
   */
  float getRandomNumberGauss(float mu = 0.0,
                             float sigma = 1.0);

  /**
     Gets a random float, using a Gaussian distribution with mu 0 and sigma 1.0

     @param mu: the mu parameter for the Gaussian distribution
     @param sigma: the sigma parameter for the Gaussian distribution
     @return returns a float from the Gaussian distribution controlled by mu and
        sigma
   */
  float getGauss(const float& mu = 0.0,
                 const float& sigma = 1.0);

  /**
     Gets an N-dimensional Vector of random Floats drawn from the uniform
        distribution set by the minRange and maxRange.

     @param numDimensions: the size of the Vector you require
     @param minRange: the minimum value in the range (inclusive)
     @param maxRange: the maximum value in the range (inclusive)
     @return returns a Vector of Floats drawn from the uniform distribution set
        by the minRange and maxRange
   */
  VectorFloat getRandomVectorUniform(uint32 numDimensions,
                                     float  minRange = 0.0,
                                     float  maxRange = 1.0);

  /**
     Gets an N-dimensional Vector of random Floats drawn from the Gaussian
        distribution controlled by mu and sigma.

     @param numDimensions: the size of the Vector you require
     @param mu: the mu parameter for the Gaussian distribution
     @param sigma: the sigma parameter for the Gaussian distribution
     @return returns a Vector of Floats drawn from the Gaussian distribution
        controlled by mu and sigma
   */
  VectorFloat getRandomVectorGauss(uint32 numDimensions,
                                   float  mu = 0.0,
                                   float  sigma = 1.0);

  /**
     Gets an N-dimensional Vector of random unsigned ints drawn from the range
        controlled by the start and end range parameters.

     @param startRange: indicates the start of the range the random subset will
        selected from (e.g. 0)
     @param endRange: indicates the end of the range the random subset will
        selected from (e.g. 100)
     @param subsetSize: controls the size of the Vector returned by the function
        (e.g. 50
     @return returns a Vector of unsigned ints selected from the
   */
  Vector<uint32>getRandomSubset(const uint32 startRange,
                                const uint32 endRange,
                                const uint32 subsetSize);

private:

  std::default_random_engine generator;
  std::uniform_real_distribution<float> uniformRealDistribution;
  std::normal_distribution<float> normalDistribution;
};
}
