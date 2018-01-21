#include "../GRT.h"
#include "DTW.h"

namespace GRT {
// Define the string that will be used to identify the object
const FString DTW::id = "DTW";
FString DTW::getId() {
  return DTW::id;
}

// Register the DTW module with the Classifier base class
RegisterClassifierModule<DTW> DTW::registerModule(DTW::getId());

DTW::DTW(bool   useScaling,
         bool   useNullRejection,
         float  nullRejectionCoeff,
         uint32 rejectionMode,
         bool   constrainWarpingPath,
         float  radius,
         bool   offsetUsingFirstSample,
         bool   useSmoothing,
         uint32 smoothingFactor,
         float  nullRejectionLikelihoodThreshold) : Classifier(DTW::getId())
{
  this->useScaling                       = useScaling;
  this->useNullRejection                 = useNullRejection;
  this->nullRejectionCoeff               = nullRejectionCoeff;
  this->nullRejectionLikelihoodThreshold = nullRejectionLikelihoodThreshold;
  this->rejectionMode                    = rejectionMode;
  this->constrainWarpingPath             = constrainWarpingPath;
  this->radius                           = radius;
  this->offsetUsingFirstSample           = offsetUsingFirstSample;
  this->useSmoothing                     = useSmoothing;
  this->smoothingFactor                  = smoothingFactor;

  supportsNullRejection = true;
  trained               = false;
  useZNormalisation     = false;
  constrainZNorm        = false;
  trimTrainingData      = false;

  zNormConstrainThreshold = 0.2;
  trimThreshold           = 0.1;
  maximumTrimPercentage   = 90;

  numTemplates   = 0;
  distanceMethod = EUCLIDEAN_DIST;

  averageTemplateLength = 0;

  classifierMode = TIMESERIES_CLASSIFIER_MODE;
}

DTW::DTW(const DTW& rhs) : Classifier(DTW::getId())
{
  *this = rhs;
}

DTW::~DTW(void) {}

DTW& DTW::operator=(const DTW& rhs) {
  if (this != &rhs) {
    this->templatesBuffer                  = rhs.templatesBuffer;
    this->distanceMatrices                 = rhs.distanceMatrices;
    this->warpPaths                        = rhs.warpPaths;
    this->continuousInputDataBuffer        = rhs.continuousInputDataBuffer;
    this->numTemplates                     = rhs.numTemplates;
    this->useSmoothing                     = rhs.useSmoothing;
    this->useZNormalisation                = rhs.useZNormalisation;
    this->constrainZNorm                   = rhs.constrainZNorm;
    this->constrainWarpingPath             = rhs.constrainWarpingPath;
    this->trimTrainingData                 = rhs.trimTrainingData;
    this->zNormConstrainThreshold          = rhs.zNormConstrainThreshold;
    this->radius                           = rhs.radius;
    this->offsetUsingFirstSample           = rhs.offsetUsingFirstSample;
    this->trimThreshold                    = rhs.trimThreshold;
    this->maximumTrimPercentage            = rhs.maximumTrimPercentage;
    this->smoothingFactor                  = rhs.smoothingFactor;
    this->distanceMethod                   = rhs.distanceMethod;
    this->rejectionMode                    = rhs.rejectionMode;
    this->nullRejectionLikelihoodThreshold = rhs.nullRejectionLikelihoodThreshold;
    this->averageTemplateLength            = rhs.averageTemplateLength;

    // Copy the classifier variables
    copyBaseVariables((Classifier *)&rhs);
  }

  return *this;
}

bool DTW::deepCopyFrom(const Classifier *classifier) {
  if (classifier == NULL) return false;

  if (this->getClassifierType() == classifier->getClassifierType()) {
    DTW *ptr = (DTW *)classifier;
    this->templatesBuffer                  = ptr->templatesBuffer;
    this->distanceMatrices                 = ptr->distanceMatrices;
    this->warpPaths                        = ptr->warpPaths;
    this->continuousInputDataBuffer        = ptr->continuousInputDataBuffer;
    this->numTemplates                     = ptr->numTemplates;
    this->useSmoothing                     = ptr->useSmoothing;
    this->useZNormalisation                = ptr->useZNormalisation;
    this->constrainZNorm                   = ptr->constrainZNorm;
    this->constrainWarpingPath             = ptr->constrainWarpingPath;
    this->trimTrainingData                 = ptr->trimTrainingData;
    this->zNormConstrainThreshold          = ptr->zNormConstrainThreshold;
    this->radius                           = ptr->radius;
    this->offsetUsingFirstSample           = ptr->offsetUsingFirstSample;
    this->trimThreshold                    = ptr->trimThreshold;
    this->maximumTrimPercentage            = ptr->maximumTrimPercentage;
    this->smoothingFactor                  = ptr->smoothingFactor;
    this->distanceMethod                   = ptr->distanceMethod;
    this->rejectionMode                    = ptr->rejectionMode;
    this->nullRejectionLikelihoodThreshold =
      ptr->nullRejectionLikelihoodThreshold;
    this->averageTemplateLength = ptr->averageTemplateLength;

    // Copy the classifier variables
    return copyBaseVariables(classifier);
  }

  return false;
}

////////////////////////// TRAINING FUNCTIONS //////////////////////////
bool DTW::train_(TimeSeriesClassificationData& data) {
  uint32 bestIndex = 0;

  // Cleanup Memory
  templatesBuffer.clear();
  classLabels.clear();
  trained = false;
  continuousInputDataBuffer.clear();

  if (trimTrainingData) {
    TimeSeriesClassificationSampleTrimmer timeSeriesTrimmer(trimThreshold,
                                                            maximumTrimPercentage);
    TimeSeriesClassificationData tempData;
    tempData.setNumDimensions(data.getNumDimensions());

    for (uint32 i = 0; i < data.getNumSamples(); i++) {
      if (timeSeriesTrimmer.trimTimeSeries(data[i])) {
        tempData.addSample(data[i].getClassLabel(), data[i].getData());
      }
      else {
        UE_LOG(GRTModule, Log,
               TEXT(
                 "Removing training sample %d from the dataset as it could not be trimmed!"),
               i);
      }
    }

    // Overwrite the original training data with the trimmed dataset
    data = tempData;
  }

  if (data.getNumSamples() == 0) {
    UE_LOG(GRTModule, Error,
           TEXT("Can't train model as there are no samples in training data!"));
    return false;
  }

  // Assign
  numClasses         = data.getNumClasses();
  numTemplates       = data.getNumClasses();
  numInputDimensions = data.getNumDimensions();
  templatesBuffer.resize(numClasses);
  classLabels.resize(numClasses);
  nullRejectionThresholds.resize(numClasses);
  averageTemplateLength = 0;

  // Need to copy the labelled training data in case we need to scale it or
  // znorm it
  TimeSeriesClassificationData trainingData(data);

  // Perform any scaling or normalization
  ranges = trainingData.getRanges();

  if (useScaling) scaleData(trainingData);

  if (useZNormalisation) znormData(trainingData);

  // For each class, run a one-to-one DTW and find the template the best
  // describes the data
  for (uint32 k = 0; k < numTemplates; k++) {
    // Get the class label for the c th class
    uint32 classLabel =
      trainingData.getClassTracker()[k].classLabel;
    TimeSeriesClassificationData classData =
      trainingData.getClassData(classLabel);
    uint32 numExamples = classData.getNumSamples();
    bestIndex = 0;

    // Set the class label of this template
    templatesBuffer[k].classLabel = classLabel;

    // Set the k th class label
    classLabels[k] = classLabel;
    UE_LOG(GRTModule, Log, TEXT("Training Template: %d Class: %d"), k,
           classLabel);

    // Check to make sure we actually have some training examples
    if (numExamples < 1) {
      UE_LOG(GRTModule, Error,
             TEXT(
               "%s::%s::%d Can not train model: Num of Example is < 1! Class: %d. Turn off null rejection if you want to use DTW with only 1 training sample per class."),
             *FString(__FILENAME__), *FString(__FUNCTION__), __LINE__,
             classLabel);
      return false;
    }

    if ((numExamples == 1) && useNullRejection) {
      UE_LOG(GRTModule, Error,
             TEXT(
               "%s::%s::%d  Can not train model as there is only 1 example in class: %d. Turn off null rejection if you want to use DTW with only 1 training sample per class."),
             *FString(__FILENAME__), *FString(__FUNCTION__), __LINE__,
             classLabel);
      return false;
    }

    if (numExamples == 1) {             // If we have just one training example
                                        // then we have to use it as the
                                        // template
      bestIndex                  = 0;
      nullRejectionThresholds[k] = 0.0; // TODO-We need a better way of
                                        // calculating this!
    }
    else {
      // Search for the best training example for this class
      if (!train_NDDTW(classData, templatesBuffer[k], bestIndex)) {
        UE_LOG(GRTModule, Error,
               TEXT(
                 "%s::%s::%d  Failed to train template for class with label: %d."),
               *FString(__FILENAME__), *FString(
                 __FUNCTION__), __LINE__, classLabel);
        return false;
      }
    }

    // Add the template with the best index to the buffer
    int trainingMethod = 0;

    if (useSmoothing) trainingMethod = 1;

    switch (trainingMethod) {
    case (0): // Standard Training
      templatesBuffer[k].timeSeries = classData[bestIndex].getData();
      break;

    case (1): // Training using Smoothing
              // Smooth the data, reducing its size by a factor set by
              // smoothFactor
      smoothData(classData[bestIndex].getData(), smoothingFactor,
                 templatesBuffer[k].timeSeries);
      break;

    default:
      UE_LOG(GRTModule, Error,
             TEXT(
               "%s::%s::%d  Can not train model: Unknown training method "),
             *FString(__FILENAME__), *FString(__FUNCTION__), __LINE__);
      return false;

      break;
    }

    if (offsetUsingFirstSample) {
      offsetTimeseries(templatesBuffer[k].timeSeries);
    }

    // Add the average length of the training examples for this template to the
    // overall averageTemplateLength
    averageTemplateLength += templatesBuffer[k].averageTemplateLength;
  }

  // Flag that the models have been trained
  trained               = true;
  converged             = true;
  averageTemplateLength = averageTemplateLength / numTemplates;

  // Recompute the null rejection thresholds
  recomputeNullRejectionThresholds();

  // Resize the prediction results to make sure it is setup for realtime
  // prediction
  continuousInputDataBuffer.clear();
  continuousInputDataBuffer.resize(averageTemplateLength,
                                   VectorFloat(numInputDimensions, 0));
  classLikelihoods.resize(numTemplates, DEFAULT_NULL_LIKELIHOOD_VALUE);
  classDistances.resize(numTemplates, 0);
  predictedClassLabel = GRT_DEFAULT_NULL_CLASS_LABEL;
  maxLikelihood       = DEFAULT_NULL_LIKELIHOOD_VALUE;

  // Training complete
  return trained;
}

bool DTW::train_NDDTW(TimeSeriesClassificationData& trainingData,
                      DTWTemplate                 & dtwTemplate,
                      uint32                      & bestIndex) {
  uint32 numExamples = trainingData.getNumSamples();
  VectorFloat results(numExamples, 0.0);
  MatrixFloat distanceResults(numExamples, numExamples);

  dtwTemplate.averageTemplateLength = 0;

  for (uint32 m = 0; m < numExamples; m++) {
    MatrixFloat templateA; // The m th template
    MatrixFloat templateB; // The n th template
    dtwTemplate.averageTemplateLength += trainingData[m].getLength();

    // Smooth the data if required
    if (useSmoothing) smoothData(
        trainingData[m].getData(), smoothingFactor, templateA);
    else templateA = trainingData[m].getData();

    if (offsetUsingFirstSample) {
      offsetTimeseries(templateA);
    }

    for (uint32 n = 0; n < numExamples; n++) {
      if (m != n) {
        // Smooth the data if required
        if (useSmoothing) smoothData(
            trainingData[n].getData(), smoothingFactor, templateB);
        else templateB = trainingData[n].getData();

        if (offsetUsingFirstSample) {
          offsetTimeseries(templateB);
        }

        // Compute the distance between the two time series
        MatrixFloat distanceMatrix(templateA.getNumRows(),
                                   templateB.getNumRows());
        Vector<IndexDist> warpPath;
        float dist = computeDistance(templateA,
                                     templateB,
                                     distanceMatrix,
                                     warpPath);
        UE_LOG(GRTModule, Log, TEXT(
                 "Template: %d  Timeseries: %d  Dist: %f"), m, n, dist);

        // Update the results values
        distanceResults[m][n] = dist;
        results[m]           += dist;
      }
      else distanceResults[m][n] = 0; // The distance is zero because the two
                                      // timeseries are the same
    }
  }

  for (uint32 m = 0; m < numExamples; m++) results[m] /= (numExamples - 1);

  // Find the best average result, this is the result with the minimum value
  bestIndex = 0;
  float bestAverage = results[0];

  for (uint32 m = 1; m < numExamples; m++) {
    if (results[m] < bestAverage) {
      bestAverage = results[m];
      bestIndex   = m;
    }
  }

  if (numExamples > 2) {
    // Work out the threshold value for the best template
    dtwTemplate.trainingMu    = results[bestIndex];
    dtwTemplate.trainingSigma = 0.0;

    for (uint32 n = 0; n < numExamples; n++) {
      if (n != bestIndex) {
        dtwTemplate.trainingSigma += SQR(
          distanceResults[bestIndex][n] - dtwTemplate.trainingMu);
      }
    }
    dtwTemplate.trainingSigma =
      sqrt(dtwTemplate.trainingSigma / float(numExamples - 2));
  }
  else {
    UE_LOG(GRTModule, Warning,
           TEXT(
             "%s::%s::%d  There are not enough examples to compute the trainingMu and trainingSigma for the template for class %d"),
           *FString(__FILENAME__), *FString(
             __FUNCTION__), __LINE__, dtwTemplate.classLabel);
    dtwTemplate.trainingMu    = 0.0;
    dtwTemplate.trainingSigma = 0.0;
  }

  // Set the average length of the training examples
  dtwTemplate.averageTemplateLength =
    (uint32)(dtwTemplate.averageTemplateLength / float(numExamples));

  UE_LOG(GRTModule, Log, TEXT(
           "AverageTemplateLength: %d"), dtwTemplate.averageTemplateLength);

  // Flag that the training was successfully
  return true;
}

bool DTW::predict_(MatrixFloat& inputTimeSeries) {
  if (!trained) {
    UE_LOG(GRTModule, Error,
           TEXT("%s::%s::%d   The DTW templates have not been trained!"),
           *FString(__FILENAME__), *FString(__FUNCTION__), __LINE__);
    return false;
  }

  if (classLikelihoods.size() != numTemplates) classLikelihoods.resize(
      numTemplates);

  if (classDistances.size() != numTemplates) classDistances.resize(numTemplates);

  predictedClassLabel = 0;
  maxLikelihood       = DEFAULT_NULL_LIKELIHOOD_VALUE;

  for (uint32 k = 0; k < classLikelihoods.size(); k++) {
    classLikelihoods[k] = 0;
    classDistances[k]   = DEFAULT_NULL_LIKELIHOOD_VALUE;
  }

  if (numInputDimensions != inputTimeSeries.getNumCols()) {
    UE_LOG(GRTModule, Error,
           TEXT(
             "%s::%s::%d  The number of features in the model (%d) do not match that of the input time series (%d)"),
           *FString(__FILENAME__), *FString(
             __FUNCTION__), __LINE__, numInputDimensions,
           inputTimeSeries.getNumCols());
    return false;
  }

  // Perform any preprocessing if required
  MatrixFloat *timeSeriesPtr = &inputTimeSeries;
  MatrixFloat  processedTimeSeries;
  MatrixFloat  tempMatrix;

  if (useScaling) {
    scaleData(*timeSeriesPtr, processedTimeSeries);
    timeSeriesPtr = &processedTimeSeries;
  }

  // Normalize the data if needed
  if (useZNormalisation) {
    znormData(*timeSeriesPtr, processedTimeSeries);
    timeSeriesPtr = &processedTimeSeries;
  }

  // Smooth the data if required
  if (useSmoothing) {
    smoothData(*timeSeriesPtr, smoothingFactor, tempMatrix);
    timeSeriesPtr = &tempMatrix;
  }

  // Offset the timeseries if required
  if (offsetUsingFirstSample) {
    offsetTimeseries(*timeSeriesPtr);
  }

  // Make the prediction by finding the closest template
  float sum = 0;

  if (distanceMatrices.size() != numTemplates) distanceMatrices.resize(
      numTemplates);

  if (warpPaths.size() != numTemplates) warpPaths.resize(numTemplates);

  // Test the timeSeries against all the templates in the timeSeries buffer
  for (uint32 k = 0; k < numTemplates; k++) {
    // Perform DTW
    classDistances[k] = computeDistance(templatesBuffer[k].timeSeries,
                                        *timeSeriesPtr,
                                        distanceMatrices[k],
                                        warpPaths[k]);

    if (classDistances[k] > 1e-8)
    {
      classLikelihoods[k] = 1.0 / classDistances[k];
    }
    else
    {
      classLikelihoods[k] = 1e8;
    }

    sum += classLikelihoods[k];
  }

  // See which gave the min distance
  uint32 closestTemplateIndex = 0;
  bestDistance = classDistances[0];

  for (uint32 k = 1; k < numTemplates; k++) {
    if (classDistances[k] < bestDistance) {
      bestDistance         = classDistances[k];
      closestTemplateIndex = k;
    }
  }

  // Normalize the class likelihoods and check which class has the maximum
  // likelihood
  uint32 maxLikelihoodIndex = 0;
  maxLikelihood = 0;

  if (sum > 0) {
    for (uint32 k = 0; k < numTemplates; k++) {
      classLikelihoods[k] /= sum;

      if (classLikelihoods[k] > maxLikelihood) {
        maxLikelihood      = classLikelihoods[k];
        maxLikelihoodIndex = k;
      }
    }
  }

  if (useNullRejection) {
    switch (rejectionMode) {
    case TEMPLATE_THRESHOLDS:

      if (bestDistance <=
          nullRejectionThresholds[closestTemplateIndex]) predictedClassLabel =
          templatesBuffer[closestTemplateIndex].classLabel;
      else predictedClassLabel = GRT_DEFAULT_NULL_CLASS_LABEL;
      break;

    case CLASS_LIKELIHOODS:

      if (maxLikelihood >=
          nullRejectionLikelihoodThreshold)  predictedClassLabel =
          templatesBuffer[maxLikelihoodIndex].classLabel;
      else predictedClassLabel = GRT_DEFAULT_NULL_CLASS_LABEL;
      break;

    case THRESHOLDS_AND_LIKELIHOODS:

      if ((bestDistance <= nullRejectionThresholds[closestTemplateIndex]) &&
          (maxLikelihood >=
           nullRejectionLikelihoodThreshold)) predictedClassLabel =
          templatesBuffer[closestTemplateIndex].classLabel;
      else predictedClassLabel = GRT_DEFAULT_NULL_CLASS_LABEL;
      break;

    default:
      UE_LOG(GRTModule, Error, TEXT(
               "%s::%s::%d  Unknown RejectionMode!"), *FString(__FILENAME__),
             *FString(__FUNCTION__), __LINE__);
      return false;

      break;
    }
  }
  else predictedClassLabel = templatesBuffer[closestTemplateIndex].classLabel;
  return true;
}

bool DTW::predict_(VectorFloat& inputVector) {
  if (!trained) {
    UE_LOG(GRTModule, Error, TEXT(
             "%s::%s::%d  The model has not been trained!"), *FString(
             __FILENAME__), *FString(__FUNCTION__), __LINE__);
    return false;
  }
  predictedClassLabel = 0;
  maxLikelihood       = DEFAULT_NULL_LIKELIHOOD_VALUE;
  std::fill(classLikelihoods.begin(),
            classLikelihoods.end(),
            DEFAULT_NULL_LIKELIHOOD_VALUE);
  std::fill(classDistances.begin(), classDistances.end(), 0);

  if (numInputDimensions != inputVector.getSize()) {
    UE_LOG(GRTModule, Error,
           TEXT(
             "%s::%s::%d  The number of features in the model %d does not match that of the input Vector %d"),
           *FString(__FILENAME__), *FString(
             __FUNCTION__), __LINE__, numInputDimensions, inputVector.size());
    return false;
  }

  // Add the new input to the circular buffer
  continuousInputDataBuffer.push_back(inputVector);

  if (continuousInputDataBuffer.getNumValuesInBuffer() < averageTemplateLength) {
    // We haven't got enough samples yet so can't do the prediction
    return true;
  }

  // Copy the data into a temporary matrix
  const uint32 M = continuousInputDataBuffer.getSize();
  const uint32 N = numInputDimensions;
  MatrixFloat  predictionTimeSeries(M, N);

  for (uint32 i = 0; i < M; i++) {
    for (uint32 j = 0; j < N; j++) {
      predictionTimeSeries[i][j] = continuousInputDataBuffer[i][j];
    }
  }

  // Run the prediction
  return predict(predictionTimeSeries);
}

bool DTW::reset() {
  continuousInputDataBuffer.clear();

  if (trained) {
    continuousInputDataBuffer.resize(averageTemplateLength,
                                     VectorFloat(numInputDimensions, 0));
    recomputeNullRejectionThresholds();
  }
  return true;
}

bool DTW::clear() {
  // Clear the Classifier variables
  Classifier::clear();

  // Clear the DTW model
  templatesBuffer.clear();
  distanceMatrices.clear();
  warpPaths.clear();
  continuousInputDataBuffer.clear();

  return true;
}

bool DTW::recomputeNullRejectionThresholds() {
  if (!trained) return false;

  // Copy the null rejection thresholds into one buffer so they can easily be
  // accessed from the base class
  nullRejectionThresholds.resize(numTemplates);

  for (uint32 k = 0; k < numTemplates; k++) {
    // The threshold is set as the mean distance plus gamma standard deviations
    nullRejectionThresholds[k] = templatesBuffer[k].trainingMu +
                                 (templatesBuffer[k].trainingSigma *
                                  nullRejectionCoeff);
  }

  return true;
}

bool DTW::setModels(Vector<DTWTemplate>newTemplates) {
  if (newTemplates.size() == templatesBuffer.size()) {
    templatesBuffer = newTemplates;

    // Make sure the class labels have not changed
    classLabels.resize(templatesBuffer.size());

    for (uint32 i = 0; i < templatesBuffer.size(); i++) {
      classLabels[i] = templatesBuffer[i].classLabel;
    }
    return true;
  }
  return false;
}

////////////////////////// computeDistance
// ///////////////////////////////////////////

float DTW::computeDistance(MatrixFloat      & timeSeriesA,
                           MatrixFloat      & timeSeriesB,
                           MatrixFloat      & distanceMatrix,
                           Vector<IndexDist>& warpPath) {
  const int M = timeSeriesA.getNumRows();
  const int N = timeSeriesB.getNumRows();
  const int C = timeSeriesA.getNumCols();
  int       i, j, k, index = 0;
  float     totalDist, v, normFactor = 0.;

  warpPath.clear();

  if ((int(distanceMatrix.getNumRows()) != M) ||
      (int(distanceMatrix.getNumCols()) != N)) {
    distanceMatrix.resize(M, N);
  }

  switch (distanceMethod) {
  case (ABSOLUTE_DIST):

    for (i = 0; i < M; i++) {
      for (j = 0; j < N; j++) {
        distanceMatrix[i][j] = 0.0;

        for (k = 0; k < C; k++) {
          distanceMatrix[i][j] += fabs(timeSeriesA[i][k] - timeSeriesB[j][k]);
        }
      }
    }
    break;

  case (EUCLIDEAN_DIST):

    // Calculate Euclidean Distance for all possible values
    for (i = 0; i < M; i++) {
      for (j = 0; j < N; j++) {
        distanceMatrix[i][j] = 0.0;

        for (k = 0; k < C; k++) {
          distanceMatrix[i][j] += SQR(timeSeriesA[i][k] - timeSeriesB[j][k]);
        }
        distanceMatrix[i][j] = sqrt(distanceMatrix[i][j]);
      }
    }
    break;

  case (NORM_ABSOLUTE_DIST):

    for (i = 0; i < M; i++) {
      for (j = 0; j < N; j++) {
        distanceMatrix[i][j] = 0.0;

        for (k = 0; k < C; k++) {
          distanceMatrix[i][j] += fabs(timeSeriesA[i][k] - timeSeriesB[j][k]);
        }
        distanceMatrix[i][j] /= N;
      }
    }
    break;

  default:
    UE_LOG(GRTModule, Error, TEXT(
             "%s::%s::%d  Unknown distance method: %d"), *FString(
             __FILENAME__), *FString(__FUNCTION__), __LINE__, distanceMethod);
    return -1;

    break;
  }

  // Run the recursive search function to build the cost matrix
  auto distance = sqrt(d(M - 1, N - 1, distanceMatrix, M, N));
  // UE_LOG(GRTModule, Log, TEXT("Distance: %f"), distance);

  if (grt_isinf(distance) || grt_isnan(distance)) {
    UE_LOG(GRTModule, Warning, TEXT(
             "%s::%s::%d  Distance Matrix Values are Inf!"), *FString(
             __FILENAME__), *FString(__FUNCTION__), __LINE__);
    return INFINITY;
  }

  // The distMatrix values are negative so make them positive
  for (i = 0; i < M; i++) {
    for (j = 0; j < N; j++) {
      distanceMatrix[i][j] = fabs(distanceMatrix[i][j]);
    }
  }

  // Now Create the Warp Path through the cost matrix, starting at the end
  i         = M - 1;
  j         = N - 1;
  totalDist = distanceMatrix[i][j];
  warpPath.push_back(IndexDist(i, j, distanceMatrix[i][j]));

  // Use dynamic programming to navigate through the cost matrix until [0][0]
  // has been reached
  normFactor = 1;

  while (true) {
    if ((i == 0) && (j == 0)) break;

    if (i == 0)   j--; else {
      if (j == 0) i--;
      else {
        // Find the minimum cell to move to
        v     = grt_numeric_limits<float>::max();
        index = 0;

        if (distanceMatrix[i - 1][j] < v) { v     = distanceMatrix[i - 1][j];
                                            index = 1; }

        if (distanceMatrix[i][j - 1] < v) { v     = distanceMatrix[i][j - 1];
                                            index = 2; }

        if (distanceMatrix[i - 1][j - 1] <= v)   index = 3;

        switch (index) {
        case (1):
          i--;
          break;

        case (2):
          j--;
          break;

        case (3):
          i--;
          j--;
          break;

        default:
          UE_LOG(GRTModule, Warning,
                 TEXT(
                   "%s::%s::%d   Could not compute a warping path for the input matrix! Dist: %f i: %d, j: %d"),
                 *FString(__FILENAME__), *FString(
                   __FUNCTION__), __LINE__, distanceMatrix[i - 1][j], i, j);
          return INFINITY;

          break;
        }
      }
    }
    normFactor++;
    totalDist += distanceMatrix[i][j];
    warpPath.push_back(IndexDist(i, j, distanceMatrix[i][j]));
  }

  return totalDist / normFactor;
}

float DTW::d(int m, int n, MatrixFloat& distanceMatrix, const int M,
             const int N) {
  float dist = 0;

  // The following is based on Matlab code by Eamonn Keogh and Michael Pazzani
  // If this cell is std::numeric_limits<float>::quiet_NaN(); then it has
  // already been flagged as unreachable
  if (grt_isnan(distanceMatrix[m][n])) {
    return NAN;
  }

  if (constrainWarpingPath) {
    auto r = FGenericPlatformMath::CeilToInt(std::min(M, N) * radius);

    // Test to see if the current cell is outside of the warping window
    float nextM = M - 1;
    float nextN = N - 1;
    auto  tmp   = n - nextN * m / nextM;

    if (fabs(tmp) > r) {
      if (tmp > 0) {
        for (uint8 i = 0; i < m; i++) {
          for (uint8 j = n; j < N; j++) {
            distanceMatrix[i][j] = NAN;
          }
        }
      }
      else {
        for (uint8 i = m; i < M; i++) {
          for (uint8 j = 0; j < n; j++) {
            distanceMatrix[i][j] = NAN;
          }
        }
      }
      return NAN;
    }
  }

  // If this cell contains a negative value then it has already been searched
  // The cost is therefore the absolute value of the negative value so return it
  if (distanceMatrix[m][n] < 0) {
    dist = fabs(distanceMatrix[m][n]);
    return dist;
  }

  // Case 1: A warping path has reached the end
  // Return the contribution of distance
  // Negate the value, to record the fact that this cell has been visited
  // End of recursion
  if ((m == 0) && (n == 0)) {
    dist                 = distanceMatrix[0][0];
    distanceMatrix[0][0] = -distanceMatrix[0][0];
    return dist;
  }

  // Case 2: we are somewhere in the top row of the matrix
  // Only need to consider moving left
  if (m == 0) {
    float contribDist = d(m, n - 1, distanceMatrix, M, N);
    dist = distanceMatrix[m][n] + contribDist;

    distanceMatrix[m][n] = -dist;
    return dist;
  }
  else {
    // Case 3: we are somewhere in the left column of the matrix
    // Only need to consider moving down
    if (n == 0) {
      float contribDist = d(m - 1, n, distanceMatrix, M, N);
      dist = distanceMatrix[m][n] + contribDist;

      distanceMatrix[m][n] = -dist;
      return dist;
    }

    else {
      // Case 4: We are somewhere away from the edges so consider moving in the
      // three main directions
      float contribDist1 = d(m - 1, n - 1, distanceMatrix, M, N);
      float contribDist2 = d(m - 1, n, distanceMatrix, M, N);
      float contribDist3 = d(m, n - 1, distanceMatrix, M, N);
      float minValue     = grt_numeric_limits<float>::max();
      uint8 index        = 0;

      if (contribDist1 < minValue) { minValue = contribDist1; index = 1; }

      if (contribDist2 < minValue) { minValue = contribDist2; index = 2; }

      if (contribDist3 < minValue) { minValue = contribDist3; index = 3; }

      switch (index) {
      case 1:
        dist = distanceMatrix[m][n] + minValue;
        break;

      case 2:
        dist = distanceMatrix[m][n] + minValue;
        break;

      case 3:
        dist = distanceMatrix[m][n] + minValue;
        break;

      default:
        break;
      }

      distanceMatrix[m][n] = -dist; // Negate the value to record that it has
                                    // been visited
      return dist;
    }
  }

  // This should not happen!
  return dist;
}

inline float DTW::MIN_(float a, float b, float c) {
  float v = a;

  if (b < v) v = b;

  if (c < v) v = c;
  return v;
}

////////////////////////// SCALING AND NORMALISATION FUNCTIONS
// //////////////////////////

void DTW::scaleData(TimeSeriesClassificationData& trainingData) {
  // Scale the data using the min and max values
  for (uint32 i = 0; i < trainingData.getNumSamples(); i++) {
    scaleData(trainingData[i].getData(), trainingData[i].getData());
  }
}

void DTW::scaleData(MatrixFloat& data, MatrixFloat& scaledData) {
  const uint32 R = data.getNumRows();
  const uint32 C = data.getNumCols();

  if ((scaledData.getNumRows() != R) || (scaledData.getNumCols() != C)) {
    scaledData.resize(R, C);
  }

  // Scale the data using the min and max values
  for (uint32 i = 0; i < R; i++)
    for (uint32 j = 0; j < C; j++) scaledData[i][j] = grt_scale(data[i][j],
                                                                ranges[j].minValue,
                                                                ranges[j].maxValue,
                                                                0.0f,
                                                                1.0f);
}

void DTW::znormData(TimeSeriesClassificationData& trainingData) {
  for (uint32 i = 0; i < trainingData.getNumSamples(); i++) {
    znormData(trainingData[i].getData(), trainingData[i].getData());
  }
}

void DTW::znormData(MatrixFloat& data, MatrixFloat& normData) {
  const uint32 R = data.getNumRows();
  const uint32 C = data.getNumCols();

  if ((normData.getNumRows() != R) || (normData.getNumCols() != C)) {
    normData.resize(R, C);
  }

  for (uint32 j = 0; j < C; j++) {
    float mean   = 0.0;
    float stdDev = 0.0;

    // Calculate Mean
    for (uint32 i = 0; i < R; i++) mean += data[i][j];
    mean /= float(R);

    // Calculate Std Dev
    for (uint32 i = 0; i < R; i++) stdDev += grt_sqr(data[i][j] - mean);
    stdDev = grt_sqrt(stdDev / (R - 1.0));

    if (constrainZNorm && (stdDev < 0.01)) {
      // Normalize the data to 0 mean
      for (uint32 i = 0; i < R; i++) normData[i][j] = (data[i][j] - mean);
    }
    else {
      // Normalize the data to 0 mean and standard deviation of 1
      for (uint32 i = 0; i < R;
           i++) normData[i][j] = (data[i][j] - mean) / stdDev;
    }
  }
}

void DTW::smoothData(VectorFloat& data,
                     uint32       smoothFactor,
                     VectorFloat& resultsData) {
  const uint32 M = (uint32)data.size();
  const uint32 N = (uint32)floor(float(M) / float(smoothFactor));

  resultsData.resize(N, 0);

  for (uint32 i = 0; i < N; i++) resultsData[i] = 0.0;

  if ((smoothFactor == 1) || (M < smoothFactor)) {
    resultsData = data;
    return;
  }

  for (uint32 i = 0; i < N; i++) {
    float  mean  = 0.0;
    uint32 index = i * smoothFactor;

    for (uint32 x = 0; x < smoothFactor; x++) {
      mean += data[index + x];
    }
    resultsData[i] = mean / smoothFactor;
  }

  // Add on the data that does not fit into the window
  if (M % smoothFactor != 0.0) {
    float mean = 0.0;

    for (uint32 i = N * smoothFactor; i < M; i++) mean += data[i];
    mean /= M - (N * smoothFactor);

    // Add one to the end of the Vector
    VectorFloat tempVector(N + 1);

    for (uint32 i = 0; i < N; i++) tempVector[i] = resultsData[i];
    tempVector[N] = mean;
    resultsData   = tempVector;
  }
}

void DTW::smoothData(MatrixFloat& data,
                     uint32       smoothFactor,
                     MatrixFloat& resultsData) {
  const uint32 M = data.getNumRows();
  const uint32 C = data.getNumCols();
  const uint32 N = (uint32)floor(float(M) / float(smoothFactor));

  resultsData.resize(N, C);

  if ((smoothFactor == 1) || (M < smoothFactor)) {
    resultsData = data;
    return;
  }

  for (uint32 i = 0; i < N; i++) {
    for (uint32 j = 0; j < C; j++) {
      float mean  = 0.0;
      int   index = i * smoothFactor;

      for (uint32 x = 0; x < smoothFactor; x++) {
        mean += data[index + x][j];
      }
      resultsData[i][j] = mean / smoothFactor;
    }
  }

  // Add on the data that does not fit into the window
  if (M % smoothFactor != 0.0) {
    VectorFloat mean(C, 0.0);

    for (uint32 j = 0; j < C; j++) {
      for (uint32 i = N * smoothFactor; i < M; i++) mean[j] += data[i][j];
      mean[j] /= M - (N * smoothFactor);
    }

    // Add one row to the end of the Matrix
    MatrixFloat tempMatrix(N + 1, C);

    for (uint32 i = 0; i < N; i++)
      for (uint32 j = 0; j < C; j++) tempMatrix[i][j] = resultsData[i][j];

    for (uint32 j = 0; j < C; j++) tempMatrix[N][j] = mean[j];
    resultsData = tempMatrix;
  }
}

bool DTW::setDistanceMethod(uint32 _distanceMethod) {
  if ((_distanceMethod == ABSOLUTE_DIST) || (_distanceMethod == EUCLIDEAN_DIST) ||
      (_distanceMethod == NORM_ABSOLUTE_DIST)) {
    this->distanceMethod = _distanceMethod;
    return true;
  }
  return false;
}

////////////////////////////// SAVE & LOAD FUNCTIONS
// ////////////////////////////////

bool DTW::save(std::fstream& file) const {
  if (!file.is_open()) {
    UE_LOG(GRTModule, Error, TEXT(
             "%s::%s::%d  Could not open file to save data"), *FString(
             __FILENAME__), *FString(__FUNCTION__), __LINE__);
    return false;
  }

  file << "GRT_DTW_Model_File_V2.0" << std::endl;

  // Write the classifier settings to the file
  if (!Classifier::saveBaseSettingsToFile(file)) {
    UE_LOG(GRTModule, Error,
           TEXT(
             "%s::%s::%d  Failed to save classifier base settings to file!"),
           *FString(__FILENAME__), *FString(__FUNCTION__), __LINE__);
    return false;
  }

  file << "DistanceMethod: ";

  switch (distanceMethod) {
  case (ABSOLUTE_DIST):
    file << ABSOLUTE_DIST << std::endl;
    break;

  case (EUCLIDEAN_DIST):
    file << EUCLIDEAN_DIST << std::endl;
    break;

  default:
    file << ABSOLUTE_DIST << std::endl;
    break;
  }
  file << "UseSmoothing: " << useSmoothing << std::endl;
  file << "SmoothingFactor: " << smoothingFactor << std::endl;
  file << "UseZNormalisation: " << useZNormalisation << std::endl;
  file << "OffsetUsingFirstSample: " << offsetUsingFirstSample << std::endl;
  file << "ConstrainWarpingPath: " << constrainWarpingPath << std::endl;
  file << "Radius: " << radius << std::endl;
  file << "RejectionMode: " << rejectionMode << std::endl;

  if (trained) {
    file << "NumberOfTemplates: " << numTemplates << std::endl;
    file << "OverallAverageTemplateLength: " << averageTemplateLength <<
      std::endl;

    // Save each template
    for (uint32 i = 0; i < numTemplates; i++) {
      file << "***************TEMPLATE***************" << std::endl;
      file << "Template: " << i + 1 << std::endl;
      file << "ClassLabel: " << templatesBuffer[i].classLabel << std::endl;
      file << "TimeSeriesLength: " <<
        templatesBuffer[i].timeSeries.getNumRows() << std::endl;
      file << "TemplateThreshold: " << nullRejectionThresholds[i] << std::endl;
      file << "TrainingMu: " << templatesBuffer[i].trainingMu << std::endl;
      file << "TrainingSigma: " << templatesBuffer[i].trainingSigma << std::endl;
      file << "AverageTemplateLength: " <<
        templatesBuffer[i].averageTemplateLength << std::endl;
      file << "TimeSeries: " << std::endl;

      for (uint32 k = 0; k < templatesBuffer[i].timeSeries.getNumRows(); k++) {
        for (uint32 j = 0; j < templatesBuffer[i].timeSeries.getNumCols(); j++) {
          file << templatesBuffer[i].timeSeries[k][j] << "\t";
        }
        file << std::endl;
      }
    }
  }

  return true;
}

bool DTW::load(std::fstream& file) {
  std::string word;
  uint32 timeSeriesLength;
  uint32 ts;

  if (!file.is_open())
  {
    UE_LOG(GRTModule, Error, TEXT("%s::%s::%d  Failed to open file!"),
           *FString(__FILENAME__), *FString(__FUNCTION__), __LINE__);
    return false;
  }

  file >> word;

  // Check to see if we should load a legacy file
  if (word == "GRT_DTW_Model_File_V1.0") {
    return loadLegacyModelFromFile(file);
  }

  // Check to make sure this is a file with the DTW File Format
  if (word != "GRT_DTW_Model_File_V2.0") {
    UE_LOG(GRTModule, Error, TEXT("%s::%s::%d  Unknown file header!"),
           *FString(__FILENAME__), *FString(__FUNCTION__), __LINE__);
    return false;
  }

  // Load the base settings from the file
  if (!Classifier::loadBaseSettingsFromFile(file)) {
    UE_LOG(GRTModule, Error,
           TEXT("%s::%s::%d  Failed to load base settings from file!"),
           *FString(__FILENAME__), *FString(__FUNCTION__), __LINE__);
    return false;
  }

  // Check and load the Distance Method
  file >> word;

  if (word != "DistanceMethod:") {
    UE_LOG(GRTModule, Error, TEXT(
             "%s::%s::%d  Failed to find DistanceMethod!"), *FString(
             __FILENAME__), *FString(__FUNCTION__), __LINE__);
    return false;
  }
  file >> distanceMethod;

  // Check and load if Smoothing is used
  file >> word;

  if (word != "UseSmoothing:") {
    UE_LOG(GRTModule, Error, TEXT(
             "%s::%s::%d  Failed to find UseSmoothing!"), *FString(
             __FILENAME__), *FString(__FUNCTION__), __LINE__);
    return false;
  }
  file >> useSmoothing;

  // Check and load what the smoothing factor is
  file >> word;

  if (word != "SmoothingFactor:") {
    UE_LOG(GRTModule, Error, TEXT(
             "%s::%s::%d  Failed to find SmoothingFactor!"), *FString(
             __FILENAME__), *FString(__FUNCTION__), __LINE__);
    return false;
  }
  file >> smoothingFactor;

  // Check and load if ZNormalization is used
  file >> word;

  if (word != "UseZNormalisation:") {
    UE_LOG(GRTModule, Error, TEXT(
             "%s::%s::%d   Failed to find UseZNormalisation!"), *FString(
             __FILENAME__), *FString(__FUNCTION__), __LINE__);
    return false;
  }
  file >> useZNormalisation;

  // Check and load if OffsetUsingFirstSample is used
  file >> word;

  if (word != "OffsetUsingFirstSample:") {
    UE_LOG(GRTModule, Error,
           TEXT("%s::%s::%d  Failed to find OffsetUsingFirstSample!"),
           *FString(__FILENAME__), *FString(__FUNCTION__), __LINE__);
    return false;
  }
  file >> offsetUsingFirstSample;

  // Check and load if ConstrainWarpingPath is used
  file >> word;

  if (word != "ConstrainWarpingPath:") {
    UE_LOG(GRTModule, Error,
           TEXT("%s::%s::%d   Failed to find ConstrainWarpingPath!"),
           *FString(__FILENAME__), *FString(__FUNCTION__), __LINE__);
    return false;
  }
  file >> constrainWarpingPath;

  // Check and load if ZNormalization is used
  file >> word;

  if (word != "Radius:") {
    UE_LOG(GRTModule, Error, TEXT("%s::%s::%d  Failed to find Radius!"),
           *FString(__FILENAME__), *FString(__FUNCTION__), __LINE__);
    return false;
  }
  file >> radius;

  // Check and load if Scaling is used
  file >> word;

  if (word != "RejectionMode:") {
    UE_LOG(GRTModule, Error, TEXT(
             "%s::%s::%d  Failed to find RejectionMode!"), *FString(
             __FILENAME__), *FString(__FUNCTION__), __LINE__);
    return false;
  }
  file >> rejectionMode;

  if (trained) {
    // Check and load the Number of Templates
    file >> word;

    if (word != "NumberOfTemplates:") {
      UE_LOG(GRTModule, Error,
             TEXT("%s::%s::%d   Failed to find NumberOfTemplates!"),
             *FString(__FILENAME__), *FString(__FUNCTION__), __LINE__);
      return false;
    }
    file >> numTemplates;

    // Check and load the overall average template length
    file >> word;

    if (word != "OverallAverageTemplateLength:") {
      UE_LOG(GRTModule, Error,
             TEXT(
               "%s::%s::%d   Failed to find OverallAverageTemplateLength!"),
             *FString(__FILENAME__), *FString(__FUNCTION__), __LINE__);
      return false;
    }
    file >> averageTemplateLength;

    // Clean and reset the memory
    templatesBuffer.resize(numTemplates);
    classLabels.resize(numTemplates);
    nullRejectionThresholds.resize(numTemplates);

    // Load each template
    for (uint32 i = 0; i < numTemplates; i++) {
      // Check we have the correct template
      file >> word;

      if (word != "***************TEMPLATE***************") {
        clear();
        UE_LOG(GRTModule, Error,
               TEXT("%s::%s::%d  Failed to find template header!"),
               *FString(__FILENAME__), *FString(__FUNCTION__), __LINE__);
        return false;
      }

      // Load the template number
      file >> word;

      if (word != "Template:") {
        clear();
        UE_LOG(GRTModule, Error,
               TEXT("%s::%s::%d  Failed to find Template Number!"),
               *FString(__FILENAME__), *FString(__FUNCTION__), __LINE__);
        return false;
      }

      // Check the template number
      file >> ts;

      if (ts != i + 1) {
        clear();
        UE_LOG(GRTModule, Error, TEXT(
                 "%s::%s::%d   Invalid Template Number: %d"), *FString(
                 __FILENAME__), *FString(__FUNCTION__), __LINE__, ts);
        return false;
      }

      // Get the class label of this template
      file >> word;

      if (word != "ClassLabel:") {
        clear();
        UE_LOG(GRTModule, Error, TEXT(
                 "%s::%s::%d  Failed to find ClassLabel!"), *FString(
                 __FILENAME__), *FString(__FUNCTION__), __LINE__);
        return false;
      }
      file >> templatesBuffer[i].classLabel;
      classLabels[i] = templatesBuffer[i].classLabel;

      // Get the time series length
      file >> word;

      if (word != "TimeSeriesLength:") {
        clear();
        UE_LOG(GRTModule, Error,
               TEXT("%s::%s::%d   Failed to find TimeSeriesLength!"),
               *FString(__FILENAME__), *FString(__FUNCTION__), __LINE__);
        return false;
      }
      file >> timeSeriesLength;

      // Resize the buffers
      templatesBuffer[i].timeSeries.resize(timeSeriesLength, numInputDimensions);

      // Get the template threshold
      file >> word;

      if (word != "TemplateThreshold:") {
        clear();
        UE_LOG(GRTModule, Error,
               TEXT("%s::%s::%d  Failed to find TemplateThreshold!"),
               *FString(__FILENAME__), *FString(__FUNCTION__), __LINE__);
        return false;
      }
      file >> nullRejectionThresholds[i];

      // Get the mu values
      file >> word;

      if (word != "TrainingMu:") {
        clear();
        UE_LOG(GRTModule, Error, TEXT(
                 "%s::%s::%d   Failed to find TrainingMu!"), *FString(
                 __FILENAME__), *FString(__FUNCTION__), __LINE__);
        return false;
      }
      file >> templatesBuffer[i].trainingMu;

      // Get the sigma values
      file >> word;

      if (word != "TrainingSigma:") {
        clear();
        UE_LOG(GRTModule, Error, TEXT(
                 "%s::%s::%d  Failed to find TrainingSigma!"), *FString(
                 __FILENAME__), *FString(__FUNCTION__), __LINE__);
        return false;
      }
      file >> templatesBuffer[i].trainingSigma;

      // Get the AverageTemplateLength value
      file >> word;

      if (word != "AverageTemplateLength:") {
        clear();
        UE_LOG(GRTModule, Error,
               TEXT("%s::%s::%d  Failed to find AverageTemplateLength!"),
               *FString(__FILENAME__), *FString(__FUNCTION__), __LINE__);
        return false;
      }
      file >> templatesBuffer[i].averageTemplateLength;

      // Get the data
      file >> word;

      if (word != "TimeSeries:") {
        clear();
        UE_LOG(GRTModule, Error,
               TEXT("%s::%s::%d  Failed to find template timeseries!"),
               *FString(__FILENAME__), *FString(__FUNCTION__), __LINE__);
        return false;
      }

      for (uint32 k = 0; k < timeSeriesLength; k++)
        for (uint32 j = 0; j < numInputDimensions;
             j++) file >> templatesBuffer[i].timeSeries[k][j];
    }

    // Resize the prediction results to make sure it is setup for realtime
    // prediction
    continuousInputDataBuffer.clear();
    continuousInputDataBuffer.resize(averageTemplateLength,
                                     VectorFloat(numInputDimensions, 0));
    maxLikelihood = DEFAULT_NULL_LIKELIHOOD_VALUE;
    bestDistance  = DEFAULT_NULL_DISTANCE_VALUE;
    classLikelihoods.resize(numClasses, DEFAULT_NULL_LIKELIHOOD_VALUE);
    classDistances.resize(numClasses, DEFAULT_NULL_DISTANCE_VALUE);
  }

  return true;
}

bool DTW::setRejectionMode(uint32 _rejectionMode) {
  if ((_rejectionMode == TEMPLATE_THRESHOLDS) ||
      (_rejectionMode == CLASS_LIKELIHOODS) ||
      (_rejectionMode == THRESHOLDS_AND_LIKELIHOODS)) {
    this->rejectionMode = _rejectionMode;
    return true;
  }
  return false;
}

bool DTW::setNullRejectionThreshold(float _nullRejectionLikelihoodThreshold)
{
  this->nullRejectionLikelihoodThreshold = _nullRejectionLikelihoodThreshold;
  return true;
}

bool DTW::setOffsetTimeseriesUsingFirstSample(bool _offsetUsingFirstSample) {
  this->offsetUsingFirstSample = _offsetUsingFirstSample;
  return true;
}

bool DTW::setContrainWarpingPath(bool _constrain) {
  this->constrainWarpingPath = _constrain;
  return true;
}

bool DTW::setWarpingRadius(float _radius) {
  this->radius = _radius;
  return true;
}

bool DTW::enableZNormalization(bool _useZNormalisation, bool _constrainZNorm) {
  this->useZNormalisation = _useZNormalisation;
  this->constrainZNorm    = _constrainZNorm;
  return true;
}

bool DTW::enableTrimTrainingData(bool  _trimTrainingData,
                                 float _trimThreshold,
                                 float _maximumTrimPercentage) {
  if ((_trimThreshold < 0) || (_trimThreshold > 1)) {
    UE_LOG(GRTModule, Warning,
           TEXT(
             "%s::%s::%d   Failed to set trimTrainingData. The trimThreshold must be in the range of [0 1]"),
           *FString(__FILENAME__), *FString(__FUNCTION__), __LINE__);
    return false;
  }

  if ((_maximumTrimPercentage < 0) || (_maximumTrimPercentage > 100)) {
    UE_LOG(GRTModule, Warning,
           TEXT(
             "%s::%s::%d   Failed to set trimTrainingData.  The maximumTrimPercentage must be a valid percentage in the range of [0 100]"),
           *FString(__FILENAME__), *FString(__FUNCTION__), __LINE__);
    return false;
  }

  this->trimTrainingData      = _trimTrainingData;
  this->trimThreshold         = _trimThreshold;
  this->maximumTrimPercentage = _maximumTrimPercentage;
  return true;
}

void DTW::offsetTimeseries(MatrixFloat& timeseries) {
  VectorFloat firstRow = timeseries.getRow(0);

  for (uint32 i = 0; i < timeseries.getNumRows(); i++) {
    for (uint32 j = 0; j < timeseries.getNumCols(); j++) {
      timeseries[i][j] -= firstRow[j];
    }
  }
}

bool DTW::loadLegacyModelFromFile(std::fstream& file) {
  std::string word;
  uint32 timeSeriesLength;
  uint32 ts;

  // Check and load the Number of Dimensions
  file >> word;

  if (word != "NumberOfDimensions:") {
    UE_LOG(GRTModule, Error, TEXT(
             "%s::%s::%d  Failed to find NumberOfDimensions!"), *FString(
             __FILENAME__), *FString(__FUNCTION__), __LINE__);
    return false;
  }
  file >> numInputDimensions;

  // Check and load the Number of Classes
  file >> word;

  if (word != "NumberOfClasses:") {
    UE_LOG(GRTModule, Error, TEXT(
             "%s::%s::%d   Failed to find NumberOfClasses!"), *FString(
             __FILENAME__), *FString(__FUNCTION__), __LINE__);
    return false;
  }
  file >> numClasses;

  // Check and load the Number of Templates
  file >> word;

  if (word != "NumberOfTemplates:") {
    UE_LOG(GRTModule, Error, TEXT(
             "%s::%s::%d  Failed to find NumberOfTemplates!"), *FString(
             __FILENAME__), *FString(__FUNCTION__), __LINE__);
    return false;
  }
  file >> numTemplates;

  // Check and load the Distance Method
  file >> word;

  if (word != "DistanceMethod:") {
    UE_LOG(GRTModule, Error, TEXT(
             "%s::%s::%d  Failed to find DistanceMethod!"), *FString(
             __FILENAME__), *FString(__FUNCTION__), __LINE__);
    return false;
  }
  file >> distanceMethod;

  // Check and load if UseNullRejection is used
  file >> word;

  if (word != "UseNullRejection:") {
    UE_LOG(GRTModule, Error, TEXT(
             "%s::%s::%d  Failed to find UseNullRejection!"), *FString(
             __FILENAME__), *FString(__FUNCTION__), __LINE__);
    return false;
  }
  file >> useNullRejection;

  // Check and load if Smoothing is used
  file >> word;

  if (word != "UseSmoothing:") {
    UE_LOG(GRTModule, Error, TEXT(
             "%s::%s::%d  Failed to find UseSmoothing!"), *FString(
             __FILENAME__), *FString(__FUNCTION__), __LINE__);
    return false;
  }
  file >> useSmoothing;

  // Check and load what the smoothing factor is
  file >> word;

  if (word != "SmoothingFactor:") {
    UE_LOG(GRTModule, Error, TEXT(
             "%s::%s::%d  Failed to find SmoothingFactor!"), *FString(
             __FILENAME__), *FString(__FUNCTION__), __LINE__);
    return false;
  }
  file >> smoothingFactor;

  // Check and load if Scaling is used
  file >> word;

  if (word != "UseScaling:") {
    UE_LOG(GRTModule, Error, TEXT(
             "%s::%s::%d  Failed to find UseScaling!"), *FString(
             __FILENAME__), *FString(__FUNCTION__), __LINE__);
    return false;
  }
  file >> useScaling;

  // Check and load if ZNormalization is used
  file >> word;

  if (word != "UseZNormalisation:") {
    UE_LOG(GRTModule, Error, TEXT(
             "%s::%s::%d  Failed to find UseZNormalisation!"), *FString(
             __FILENAME__), *FString(__FUNCTION__), __LINE__);
    return false;
  }
  file >> useZNormalisation;

  // Check and load if OffsetUsingFirstSample is used
  file >> word;

  if (word != "OffsetUsingFirstSample:") {
    UE_LOG(GRTModule, Error,
           TEXT("%s::%s::%d  Failed to find OffsetUsingFirstSample!"),
           *FString(__FILENAME__), *FString(__FUNCTION__), __LINE__);
    return false;
  }
  file >> offsetUsingFirstSample;

  // Check and load if ConstrainWarpingPath is used
  file >> word;

  if (word != "ConstrainWarpingPath:") {
    UE_LOG(GRTModule, Error,
           TEXT("%s::%s::%d  Failed to find ConstrainWarpingPath!"),
           *FString(__FILENAME__), *FString(__FUNCTION__), __LINE__);
    return false;
  }
  file >> constrainWarpingPath;

  // Check and load if ZNormalization is used
  file >> word;

  if (word != "Radius:") {
    UE_LOG(GRTModule, Error, TEXT("%s::%s::%d  Failed to find Radius!"),
           *FString(__FILENAME__), *FString(__FUNCTION__), __LINE__);
    return false;
  }
  file >> radius;

  // Check and load if Scaling is used
  file >> word;

  if (word != "RejectionMode:") {
    UE_LOG(GRTModule, Error, TEXT(
             "%s::%s::%d  Failed to find RejectionMode!"), *FString(
             __FILENAME__), *FString(__FUNCTION__), __LINE__);
    return false;
  }
  file >> rejectionMode;

  // Check and load gamma
  file >> word;

  if (word != "NullRejectionCoeff:") {
    UE_LOG(GRTModule, Error, TEXT(
             "%s::%s::%d  Failed to find NullRejectionCoeff!"), *FString(
             __FILENAME__), *FString(__FUNCTION__), __LINE__);
    return false;
  }
  file >> nullRejectionCoeff;

  // Check and load the overall average template length
  file >> word;

  if (word != "OverallAverageTemplateLength:") {
    UE_LOG(GRTModule, Error,
           TEXT("%s::%s::%d  Failed to find OverallAverageTemplateLength!"),
           *FString(__FILENAME__), *FString(__FUNCTION__), __LINE__);
    return false;
  }
  file >> averageTemplateLength;

  // Clean and reset the memory
  templatesBuffer.resize(numTemplates);
  classLabels.resize(numTemplates);
  nullRejectionThresholds.resize(numTemplates);

  // Load each template
  for (uint32 i = 0; i < numTemplates; i++) {
    // Check we have the correct template
    file >> word;

    while (word != "Template:") {
      file >> word;
    }
    file >> ts;

    // Check the template number
    if (ts != i + 1) {
      numTemplates = 0;
      trained      = false;
      UE_LOG(GRTModule, Error,
             TEXT("%s::%s::%d  Failed to find Invalid Template Number!"),
             *FString(__FILENAME__), *FString(__FUNCTION__), __LINE__);
      return false;
    }

    // Get the class label of this template
    file >> word;

    if (word != "ClassLabel:") {
      numTemplates = 0;
      trained      = false;
      UE_LOG(GRTModule, Error, TEXT(
               "%s::%s::%d  Failed to find ClassLabel!"), *FString(
               __FILENAME__), *FString(__FUNCTION__), __LINE__);
      return false;
    }
    file >> templatesBuffer[i].classLabel;
    classLabels[i] = templatesBuffer[i].classLabel;

    // Get the time series length
    file >> word;

    if (word != "TimeSeriesLength:") {
      numTemplates = 0;
      trained      = false;
      UE_LOG(GRTModule, Error, TEXT(
               "%s::%s::%d  Failed to find TimeSeriesLength!"), *FString(
               __FILENAME__), *FString(__FUNCTION__), __LINE__);
      return false;
    }
    file >> timeSeriesLength;

    // Resize the buffers
    templatesBuffer[i].timeSeries.resize(timeSeriesLength, numInputDimensions);

    // Get the template threshold
    file >> word;

    if (word != "TemplateThreshold:") {
      numTemplates = 0;
      trained      = false;
      UE_LOG(GRTModule, Error,
             TEXT("%s::%s::%d  Failed to find TemplateThreshold!"),
             *FString(__FILENAME__), *FString(__FUNCTION__), __LINE__);
      return false;
    }
    file >> nullRejectionThresholds[i];

    // Get the mu values
    file >> word;

    if (word != "TrainingMu:") {
      numTemplates = 0;
      trained      = false;
      UE_LOG(GRTModule, Error, TEXT(
               "%s::%s::%d  Failed to find TrainingMu!"), *FString(
               __FILENAME__), *FString(__FUNCTION__), __LINE__);
      return false;
    }
    file >> templatesBuffer[i].trainingMu;

    // Get the sigma values
    file >> word;

    if (word != "TrainingSigma:") {
      numTemplates = 0;
      trained      = false;
      UE_LOG(GRTModule, Error, TEXT(
               "%s::%s::%d  Failed to find TrainingSigma!"), *FString(
               __FILENAME__), *FString(__FUNCTION__), __LINE__);
      return false;
    }
    file >> templatesBuffer[i].trainingSigma;

    // Get the AverageTemplateLength value
    file >> word;

    if (word != "AverageTemplateLength:") {
      numTemplates = 0;
      trained      = false;
      UE_LOG(GRTModule, Error,
             TEXT("%s::%s::%d  Failed to find AverageTemplateLength!"),
             *FString(__FILENAME__), *FString(__FUNCTION__), __LINE__);
      return false;
    }
    file >> templatesBuffer[i].averageTemplateLength;

    // Get the data
    file >> word;

    if (word != "TimeSeries:") {
      numTemplates = 0;
      trained      = false;
      UE_LOG(GRTModule, Error,
             TEXT("%s::%s::%d  Failed to find template timeseries!"),
             *FString(__FILENAME__), *FString(__FUNCTION__), __LINE__);
      return false;
    }

    for (uint32 k = 0; k < timeSeriesLength; k++)
      for (uint32 j = 0; j < numInputDimensions;
           j++) file >> templatesBuffer[i].timeSeries[k][j];

    // Check for the footer
    file >> word;

    if (word != "***************************") {
      numTemplates       = 0;
      numClasses         = 0;
      numInputDimensions = 0;
      trained            = false;
      UE_LOG(GRTModule, Error, TEXT(
               "%s::%s::%d  Failed to find template footer!"), *FString(
               __FILENAME__), *FString(__FUNCTION__), __LINE__);
      return false;
    }
  }

  // Resize the prediction results to make sure it is setup for realtime
  // prediction
  continuousInputDataBuffer.clear();
  continuousInputDataBuffer.resize(averageTemplateLength,
                                   VectorFloat(numInputDimensions, 0));
  maxLikelihood = DEFAULT_NULL_LIKELIHOOD_VALUE;
  bestDistance  = DEFAULT_NULL_DISTANCE_VALUE;
  classLikelihoods.resize(numClasses, DEFAULT_NULL_LIKELIHOOD_VALUE);
  classDistances.resize(numClasses, DEFAULT_NULL_DISTANCE_VALUE);
  trained = true;
  return true;
}
}
