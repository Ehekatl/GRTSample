#pragma once

#include "../GRT.h"


namespace GRT {
// forward class definition
class VectorFloat;

class TestInstanceResult {
public:

  /**
     Default Constructor.

     Initializes the TrainingResult instance.
   */
  TestInstanceResult() {
    testMode                       = CLASSIFICATION_MODE;
    testIteration                  = 0;
    classLabel                     = 0;
    predictedClassLabel            = 0;
    unProcessedPredictedClassLabel = 0;
  }

  /**
     Copy Constructor.

     Initializes this instance by copying the data from the rhs instance

     @param const TestInstanceResult &rhs: another instance of the
        TestInstanceResult class
   */
  TestInstanceResult(const TestInstanceResult& rhs) {
    *this = rhs;
  }

  /**
     Default Destructor.
   */
  ~TestInstanceResult() {}

  /**
     Defines the Equals Operator.

     This copies the data from the rhs instance to this instance, returning a
        reference to the current instance.

     @param const TestInstanceResult &rhs: another instance of the
        TestInstanceResult class
   */
  TestInstanceResult& operator=(const TestInstanceResult& rhs) {
    if (this != &rhs) {
      this->testMode                       = rhs.testMode;
      this->testIteration                  = rhs.testIteration;
      this->classLabel                     = rhs.classLabel;
      this->predictedClassLabel            = rhs.predictedClassLabel;
      this->unProcessedPredictedClassLabel = rhs.unProcessedPredictedClassLabel;
      this->classLikelihoods               = rhs.classLikelihoods;
      this->classDistances                 = rhs.classDistances;
      this->regressionData                 = rhs.regressionData;
      this->targetData                     = rhs.targetData;
    }
    return *this;
  }

  /**
     Sets the training result for classification data. This will place the
        training mode into CLASSIFICATION_MODE.

     @param trainingIteration: the current training iteration (or epoch)
     @param accuracy: the accuracy for the current training iteration
     @return returns true if the training result was set successfully
   */
  bool setClassificationResult(const uint32       _testIteration,
                               const uint32       _classLabel,
                               const uint32       _predictedClassLabel,
                               const uint32       _unProcessedPredictedClassLabel,
                               const VectorFloat& _classLikelihoods,
                               const VectorFloat& _classDistances) {
    this->testMode                       = CLASSIFICATION_MODE;
    this->testIteration                  = _testIteration;
    this->classLabel                     = _classLabel;
    this->predictedClassLabel            = _predictedClassLabel;
    this->unProcessedPredictedClassLabel = _unProcessedPredictedClassLabel;
    this->classLikelihoods               = _classLikelihoods;
    this->classDistances                 = _classDistances;
    return true;
  }

  /**
     Sets the training result for regression data. This will place the training
        mode into REGRESSION_MODE.

     @param trainingIteration: the current training iteration (or epoch)
     @param totalSquaredTrainingError: the total squared training error for the
        current iteration
     @param rootMeanSquaredTrainingError: the root mean squared training error
        for the current iteration
     @return returns true if the training result was set successfully
   */
  bool setRegressionResult(const uint32       _testIteration,
                           const VectorFloat& _regressionData,
                           const VectorFloat& _targetData) {
    this->testMode       = REGRESSION_MODE;
    this->testIteration  = _testIteration;
    this->regressionData = _regressionData;
    this->targetData     = _targetData;
    return true;
  }

  /**
     Gets the current test mode, this will be one of the TestMode enums.

     @return returns the current test mode, this will be one of the TestMode
        enums
   */
  uint32 getTestMode() const {
    return testMode;
  }

  /**
     Gets the test iteration, this represents which test example the test
        results correspond to.

     @return returns the test iteration
   */
  uint32 getTestIteration() const {
    return testIteration;
  }

  /**
     Gets the class label. This is only useful in CLASSIFICATION_MODE.

     @return returns the class label
   */
  uint32 getClassLabel() const {
    return classLabel;
  }

  /**
     Gets the predicted class label. This is only useful in CLASSIFICATION_MODE.

     @return returns the predicted class label
   */
  uint32 getPredictedClassLabel() const {
    return predictedClassLabel;
  }

  /**
     Gets the maximum likelihood. This is only useful in CLASSIFICATION_MODE.

     @return returns the maximum likelihood
   */
  float getMaximumLikelihood() const {
    float maxLikelihood = 0;

    for (size_t i = 0; i < classLikelihoods.size(); i++) {
      if (classLikelihoods[i] > maxLikelihood) {
        maxLikelihood = classLikelihoods[i];
      }
    }
    return maxLikelihood;
  }

  /**
     Gets the squared error between the regressionData and the target data. This
        is only useful in REGRESSION_MODE.

     @return returns the squared error between the regression estimate and the
        target data
   */
  float getSquaredError() const {
    float sum = 0;

    if (regressionData.size() != targetData.size()) return 0;

    for (size_t i = 0; i < regressionData.size(); i++) {
      sum += (regressionData[i] - targetData[i]) *
             (regressionData[i] - targetData[i]);
    }
    return sum;
  }

  /**
     Gets the class likelihoods vector. This is only useful in
        CLASSIFICATION_MODE.

     @return returns the class likelihoods vector
   */
  VectorFloat getClassLikelihoods() const {
    return classLikelihoods;
  }

  /**
     Gets the class distances vector. This is only useful in
        CLASSIFICATION_MODE.

     @return returns the class distances vector
   */
  VectorFloat getDistances() const {
    return classDistances;
  }

protected:

  uint32 testMode;
  uint32 testIteration;
  uint32 classLabel;
  uint32 predictedClassLabel;
  uint32 unProcessedPredictedClassLabel;
  VectorFloat classLikelihoods;
  VectorFloat classDistances;
  VectorFloat regressionData;
  VectorFloat targetData;

public:

  enum TestMode { CLASSIFICATION_MODE = 0, REGRESSION_MODE };
};
}
