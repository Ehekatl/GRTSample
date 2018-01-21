#include "../GRT.h"
#include "Random.h"

namespace GRT {
Random::Random() : uniformRealDistribution(0.0, 1.0),
  normalDistribution(0.0, 1.0) {}

Random::~Random() {}

int Random::getRandomNumberInt(int minRange, int maxRange) {
  std::uniform_int_distribution<int> uniformIntDistribution(minRange,
                                                            maxRange - 1);

  return uniformIntDistribution(generator);
}

int Random::getRandomNumberWeighted(const Vector<int>& values,
                                    const VectorFloat& weights) {
  if (values.size() != weights.size()) return 0;

  uint32 N = (uint32)values.size();
  Vector<IndexedDouble> weightedValues(N);

  for (uint32 i = 0; i < N; i++) {
    weightedValues[i].index = values[i];
    weightedValues[i].value = weights[i];
  }

  return getRandomNumberWeighted(weightedValues);
}

int Random::getRandomNumberWeighted(Vector<IndexedDouble>weightedValues) {
  uint32 N = (uint32)weightedValues.size();

  if (N == 0) return 0;

  if (N == 1) return weightedValues[0].index;

  // Sort the weighted values by value in ascending order (so the least likely
  // value is first, the second most likely is second, etc...
  sort(weightedValues.begin(),
       weightedValues.end(), IndexedDouble::sortIndexedDoubleByValueAscending);

  // Create the accumulated sum lookup table
  Vector<float> x(N);
  x[0] = weightedValues[0].value;

  for (uint32 i = 1; i < N; i++) {
    x[i] = x[i - 1] + weightedValues[i].value;
  }

  // Generate a random value between min and the max weighted float values
  float randValue = getUniform(0.0, x[N - 1]);

  // Find which bin the rand value falls into, return the index of that bin
  for (uint32 i = 0; i < N; i++) {
    if (randValue <= x[i]) {
      return weightedValues[i].index;
    }
  }
  return 0;
}

int Random::getRandomNumberWeighted(Vector<IndexedDouble>& weightedValues,
                                    VectorFloat          & x) {
  uint32 N = (uint32)weightedValues.size();

  if (weightedValues.size() != x.size()) return 0;

  // Generate a random value between min and the max weighted float values
  float randValue = getUniform(0.0, x[N - 1]);

  // Find which bin the rand value falls into, return the index of that bin
  for (uint32 i = 0; i < N; i++) {
    if (randValue <= x[i]) {
      return weightedValues[i].index;
    }
  }
  return 0;
}

float Random::getRandomNumberUniform(float minRange, float maxRange) {
  return getUniform(minRange, maxRange);
}

float Random::getUniform(const float& minRange, const float& maxRange) {
  float r = uniformRealDistribution(generator);

  return (r * (maxRange - minRange)) + minRange;
}

float Random::getRandomNumberGauss(float mu, float sigma) {
  return getGauss(mu, sigma);
}

float Random::getGauss(const float& mu, const float& sigma) {
  return mu + (normalDistribution(generator) * sigma);
}

VectorFloat Random::getRandomVectorUniform(uint32 numDimensions,
                                           float  minRange,
                                           float  maxRange) {
  VectorFloat randomValues(numDimensions);

  for (uint32 i = 0; i < numDimensions; i++) {
    randomValues[i] = getRandomNumberUniform(minRange, maxRange);
  }
  return randomValues;
}

VectorFloat Random::getRandomVectorGauss(uint32 numDimensions,
                                         float  mu,
                                         float  sigma) {
  VectorFloat randomValues(numDimensions);

  for (uint32 i = 0; i < numDimensions; i++) {
    randomValues[i] = getRandomNumberGauss(mu, sigma);
  }
  return randomValues;
}

Vector<uint32>Random::getRandomSubset(const uint32 startRange,
                                      const uint32 endRange,
                                      const uint32 subsetSize) {
  uint32 i               = 0;
  const uint32 rangeSize = endRange - startRange;

  if (rangeSize > 0) {
    UE_LOG(GRTModule, Error,
           TEXT(
             "Random::getRandomSubset(const uint32 startRange, const uint32 endRange, const uint32 subsetSize) -- rangeSize should > 0"));
    return NULL;
  }

  if (endRange > startRange) {
    UE_LOG(GRTModule, Error,
           TEXT(
             "Random::getRandomSubset(const uint32 startRange, const uint32 endRange, const uint32 subsetSize) -- endRange should > startRange"));
    return NULL;
  }

  if (subsetSize <= rangeSize) {
    UE_LOG(GRTModule, Error,
           TEXT(
             "Random::getRandomSubset(const uint32 startRange, const uint32 endRange, const uint32 subsetSize) -- subsetSize should <= rangeSize"));
    return NULL;
  }

  Vector<uint32> indexs(rangeSize);
  Vector<uint32> subset(subsetSize);

  // Fill up the range buffer and the randomly suffle it
  for (i = startRange; i < endRange; i++) {
    indexs[i] = i;
  }
  std::random_shuffle(indexs.begin(), indexs.end());

  // Select the first X values from the randomly shuffled range buffer as the
  // subset
  for (i = 0; i < subsetSize; i++) {
    subset[i] = indexs[i];
  }

  return subset;
}
}
