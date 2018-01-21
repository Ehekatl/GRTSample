#pragma once

#include "VectorFloat.h"
#include "MatrixFloat.h"

namespace GRT {
  class GRT_API TimeSeriesClassificationSample {
  public:
    TimeSeriesClassificationSample();
    TimeSeriesClassificationSample(const uint32 classLabel, const MatrixFloat &data);
    TimeSeriesClassificationSample(const TimeSeriesClassificationSample &rhs);
    ~TimeSeriesClassificationSample();

    TimeSeriesClassificationSample& operator= (const TimeSeriesClassificationSample &rhs) {
      if (this != &rhs) {
        this->classLabel = rhs.classLabel;
        this->data = rhs.data;
      }
      return *this;
    }

    inline float* operator[] (const uint32 &n) {
      return data[n];
    }

    inline const float* operator[] (const uint32 &n) const {
      return data[n];
    }

    bool clear();
    bool addSample(const uint32 classLabel, const VectorFloat &sample);
    bool setTrainingSample(const uint32 classLabel, const MatrixFloat &data);
    inline uint32 getLength() const { return data.getNumRows(); }
    inline uint32 getNumDimensions() const { return data.getNumCols(); }
    inline uint32 getClassLabel() const { return classLabel; }
    MatrixFloat &getData() { return data; }
    const MatrixFloat &getData() const { return data; }

  protected:
    uint32 classLabel;
    MatrixFloat data;
  };
}