#include "../GRT.h"
#include "TimeSeriesClassificationSampleTrimmer.h"

namespace GRT {
TimeSeriesClassificationSampleTrimmer::TimeSeriesClassificationSampleTrimmer(
  float trimThreshold,
  float maximumTrimPercentage) {
  this->trimThreshold         = trimThreshold;
  this->maximumTrimPercentage = maximumTrimPercentage;
}

TimeSeriesClassificationSampleTrimmer::~TimeSeriesClassificationSampleTrimmer() {}

bool TimeSeriesClassificationSampleTrimmer::trimTimeSeries(
  TimeSeriesClassificationSample& timeSeries) {
  const uint32 M = timeSeries.getLength();
  const uint32 N = timeSeries.getNumDimensions();

  if (M == 0) {
    UE_LOG(GRTModule, Warning,
           TEXT(
             "trimTimeSeries(TimeSeriesClassificationSample &timeSeries) - can't trim data, the length of the input time series is 0!"));
    return false;
  }

  if (N == 0) {
    UE_LOG(GRTModule, Warning,
           TEXT(
             "trimTimeSeries(TimeSeriesClassificationSample &timeSeries) - can't trim data, the number of dimensions in the input time series is 0!"));
    return false;
  }

  // Compute the energy of the time series
  float maxValue = 0;
  VectorFloat x(M, 0);

  for (uint32 i = 1; i < M; i++) {
    for (uint32 j = 0; j < N; j++) {
      x[i] += fabs(timeSeries[i][j] - timeSeries[i - 1][j]);
    }
    x[i] /= N;

    if (x[i] > maxValue) maxValue = x[i];
  }

  // Normalize x so that the maximum energy has a value of 1
  // At the same time search for the first time x[i] passes the trim threshold
  uint32 firstIndex = 0;

  for (uint32 i = 1; i < M; i++) {
    x[i] /= maxValue;

    if ((x[i] > trimThreshold) && (firstIndex == 0)) {
      firstIndex = i;
    }
  }

  // Search for the last time x[i] passes the trim threshold
  uint32 lastIndex = 0;

  for (uint32 i = M - 1; i > firstIndex; i--) {
    if ((x[i] > trimThreshold) && (lastIndex == 0)) {
      lastIndex = i;
      break;
    }
  }

  if ((firstIndex == 0) && (lastIndex == 0)) {
    UE_LOG(GRTModule, Warning,
           TEXT("Failed to find either the first index or the last index!"));
    return false;
  }

  if (firstIndex == lastIndex) {
    UE_LOG(GRTModule, Warning, TEXT(
             "The first index and last index are the same!"));
    return false;
  }

  if (firstIndex > lastIndex) {
    UE_LOG(GRTModule, Warning,
           TEXT("The first index is greater than the last index!"));
    return false;
  }

  if (lastIndex == 0) {
    UE_LOG(GRTModule, Warning, TEXT("Failed to find the last index!"));
    lastIndex = M - 1;
  }

  // Compute how long the new time series would be if we trimmed it
  uint32 newM           = lastIndex - firstIndex;
  float  trimPercentage = (float(newM) / float(M)) * 100.0;

  if (100 - trimPercentage <= maximumTrimPercentage) {
    MatrixFloat newTimeSeries(newM, N);
    uint32 index = 0;

    for (uint32 i = firstIndex; i < lastIndex; i++) {
      for (uint32 j = 0; j < N; j++) {
        newTimeSeries[index][j] = timeSeries[i][j];
      }
      index++;
    }

    timeSeries.setTrainingSample(timeSeries.getClassLabel(), newTimeSeries);
    return true;
  }

  UE_LOG(GRTModule, Warning,
         TEXT(
           "Maximum Trim Percentage Excedded, Can't Trim Sample! Original Timeseries Length: %d Trimmed Timeseries Length: %f Percentage: %f MaximumTrimPercentage: %d"), M, newM,
         (100 - trimPercentage), maximumTrimPercentage);
  return false;
}
}
