#include "../GRT.h"
#include "TimeSeriesClassificationSample.h"

namespace GRT {
// Constructors and Destructors
TimeSeriesClassificationSample::TimeSeriesClassificationSample() : classLabel(0) {}

TimeSeriesClassificationSample::TimeSeriesClassificationSample(
  const uint32       classLabel,
  const MatrixFloat& data) {
  this->classLabel = classLabel;
  this->data       = data;
}

TimeSeriesClassificationSample::TimeSeriesClassificationSample(
  const TimeSeriesClassificationSample& rhs) {
  this->classLabel = rhs.classLabel;
  this->data       = rhs.data;
}

TimeSeriesClassificationSample::~TimeSeriesClassificationSample() {}

bool TimeSeriesClassificationSample::clear() {
  classLabel = 0;
  data.clear();
  return true;
}

bool TimeSeriesClassificationSample::addSample(const uint32       l_classLabel,
                                               const VectorFloat& sample) {
  this->classLabel = l_classLabel;
  this->data.push_back(sample);
  return true;
}

bool TimeSeriesClassificationSample::setTrainingSample(
  const uint32       l_classLabel,
  const MatrixFloat& l_data)
{
  this->classLabel = l_classLabel;
  this->data       = l_data;
  return true;
}
}
