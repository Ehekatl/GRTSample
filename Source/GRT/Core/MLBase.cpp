#include "GRT.h"
#include "MLBase.h"

namespace GRT {
MLBase::MLBase(const FString& id, const BaseType type) : GRTBase(id) {
  baseType                  = type;
  trained                   = false;
  converged                 = false;
  useScaling                = false;
  inputType                 = DATA_TYPE_UNKNOWN;
  outputType                = DATA_TYPE_UNKNOWN;
  numInputDimensions        = 0;
  numOutputDimensions       = 0;
  minNumEpochs              = 0;
  maxNumEpochs              = 100;
  batchSize                 = 1;
  numRestarts               = 1;
  validationSetSize         = 20;
  validationSetAccuracy     = 0;
  minChange                 = 1.0e-5;
  learningRate              = 0.1;
  useValidationSet          = false;
  randomiseTrainingOrder    = true;
  rmsTrainingError          = 0;
  rmsValidationError        = 0;
  totalSquaredTrainingError = 0;
}

MLBase::~MLBase(void) {
  clear();
}

bool MLBase::copyMLBaseVariables(const MLBase *mlBase) {
  if (mlBase == NULL) {
    UE_LOG(GRTModule, Error,
           TEXT("copyMLBaseVariables(MLBase *mlBase) - mlBase pointer is NULL!"));
    return false;
  }

  if (!copyGRTBaseVariables(mlBase)) {
    UE_LOG(GRTModule, Error,
           TEXT(
             "copyMLBaseVariables(MLBase *mlBase) - Failed to copy GRT Base variables!"));
    return false;
  }

  this->trained                         = mlBase->trained;
  this->converged                       = mlBase->converged;
  this->useScaling                      = mlBase->useScaling;
  this->baseType                        = mlBase->baseType;
  this->inputType                       = mlBase->inputType;
  this->outputType                      = mlBase->outputType;
  this->numInputDimensions              = mlBase->numInputDimensions;
  this->numOutputDimensions             = mlBase->numOutputDimensions;
  this->minNumEpochs                    = mlBase->minNumEpochs;
  this->maxNumEpochs                    = mlBase->maxNumEpochs;
  this->batchSize                       = mlBase->batchSize;
  this->numRestarts                     = mlBase->numRestarts;
  this->validationSetSize               = mlBase->validationSetSize;
  this->validationSetAccuracy           = mlBase->validationSetAccuracy;
  this->validationSetPrecision          = mlBase->validationSetPrecision;
  this->validationSetRecall             = mlBase->validationSetRecall;
  this->minChange                       = mlBase->minChange;
  this->learningRate                    = mlBase->learningRate;
  this->rmsTrainingError                = mlBase->rmsTrainingError;
  this->rmsValidationError              = mlBase->rmsValidationError;
  this->totalSquaredTrainingError       = mlBase->totalSquaredTrainingError;
  this->useValidationSet                = mlBase->useValidationSet;
  this->randomiseTrainingOrder          = mlBase->randomiseTrainingOrder;
  this->numTrainingIterationsToConverge = mlBase->numTrainingIterationsToConverge;
  this->trainingResults                 = mlBase->trainingResults;
  this->trainingResultsObserverManager  = mlBase->trainingResultsObserverManager;
  this->testResultsObserverManager      = mlBase->testResultsObserverManager;
  return true;
}

bool MLBase::train(TimeSeriesClassificationData trainingData) {
  return train_(trainingData);
}

bool MLBase::train_(TimeSeriesClassificationData& trainingData) {
  return false;
}

bool MLBase::train(MatrixFloat data) {
  return train_(data);
}

bool MLBase::train_(MatrixFloat& data) {
  return false;
}

bool MLBase::predict(VectorFloat inputVector) {
  return predict_(inputVector);
}

bool MLBase::predict_(VectorFloat& inputVector) {
  return false;
}

bool MLBase::predict(MatrixFloat inputMatrix) {
  return predict_(inputMatrix);
}

bool MLBase::predict_(MatrixFloat& inputMatrix) {
  return false;
}

bool MLBase::map(VectorFloat inputVector) {
  return map_(inputVector);
}

bool MLBase::map_(VectorFloat& inputVector) {
  return false;
}

bool MLBase::reset() {
  return true;
}

bool MLBase::clear() {
  trained                         = false;
  converged                       = false;
  numInputDimensions              = 0;
  numOutputDimensions             = 0;
  numTrainingIterationsToConverge = 0;
  rmsTrainingError                = 0;
  rmsValidationError              = 0;
  totalSquaredTrainingError       = 0;
  trainingResults.clear();
  validationSetPrecision.clear();
  validationSetRecall.clear();
  validationSetAccuracy = 0;
  return true;
}

bool MLBase::save(const FString& filename) const {
  std::fstream file;

  file.open(TCHAR_TO_UTF8(*filename), std::ios::out);

  if (!save(file))
  {
    return false;
  }
  file.close();
  return true;
}

bool MLBase::save(std::fstream& file) const {
  return false; // The base class returns false, as this should be overwritten
                // by the inheriting class
}

bool MLBase::load(const FString& filename) {
  std::fstream file;

  file.open(TCHAR_TO_UTF8(*filename), std::ios::in);

  if (!load(file)) {
    return false;
  }

  // Close the file
  file.close();
  return true;
}

bool MLBase::load(std::fstream& file) {
  return false; // The base class returns false, as this should be overwritten
                // by the inheriting class
}

bool MLBase::getModel(std::ostream& stream) const {
  return true;
}

FString MLBase::getModelAsString() const {
  std::stringstream stream;

  if (getModel(stream)) {
    return FString(stream.str().c_str());
  }
  return "";
}

DataType MLBase::getInputType() const {
  return inputType;
}

DataType MLBase::getOutputType() const {
  return outputType;
}

MLBase::BaseType MLBase::getType() const {
  return baseType;
}

uint32 MLBase::getNumInputDimensions() const {
  return numInputDimensions;
}

uint32 MLBase::getNumOutputDimensions() const {
  return numOutputDimensions;
}

uint32 MLBase::getNumTrainingIterationsToConverge() const {
  if (trained) {
    return numTrainingIterationsToConverge;
  }
  return 0;
}

uint32 MLBase::getMinNumEpochs() const {
  return minNumEpochs;
}

uint32 MLBase::getMaxNumEpochs() const {
  return maxNumEpochs;
}

uint32 MLBase::getBatchSize() const {
  return batchSize;
}

uint32 MLBase::getNumRestarts() const {
  return numRestarts;
}

uint32 MLBase::getValidationSetSize() const {
  return validationSetSize;
}

float MLBase::getLearningRate() const {
  return learningRate;
}

float MLBase::getRMSTrainingError() const {
  return rmsTrainingError;
}

float MLBase::getTotalSquaredTrainingError() const {
  return totalSquaredTrainingError;
}

float MLBase::getRMSValidationError() const {
  return rmsValidationError;
}

float MLBase::getValidationSetAccuracy() const {
  return validationSetAccuracy;
}

VectorFloat MLBase::getValidationSetPrecision() const {
  return validationSetPrecision;
}

VectorFloat MLBase::getValidationSetRecall() const {
  return validationSetRecall;
}

bool MLBase::getTrained() const {
  return trained;
}

bool MLBase::getConverged() const {
  return converged;
}

bool MLBase::getScalingEnabled() const {
  return useScaling;
}

bool MLBase::getIsBaseTypeClassifier() const {
  return baseType == CLASSIFIER;
}

bool MLBase::getIsBaseTypeRegressifier() const {
  return baseType == REGRESSIFIER;
}

bool MLBase::getIsBaseTypeClusterer() const {
  return baseType == CLUSTERER;
}

bool MLBase::enableScaling(bool _useScaling) {
  this->useScaling = _useScaling; return true;
}

bool MLBase::getUseValidationSet() const {
  return useValidationSet;
}

bool MLBase::setMaxNumEpochs(const uint32 _maxNumEpochs) {
  if (_maxNumEpochs == 0) {
    UE_LOG(GRTModule, Warning,
           TEXT(
             "setMaxNumEpochs(const uint32 maxNumEpochs) - The maxNumEpochs must be greater than 0!"));
    return false;
  }
  this->maxNumEpochs = _maxNumEpochs;
  return true;
}

bool MLBase::setMinNumEpochs(const uint32 _minNumEpochs) {
  this->minNumEpochs = _minNumEpochs;
  return true;
}

bool MLBase::setBatchSize(const uint32 _batchSize) {
  this->batchSize = _batchSize;
  return true;
}

bool MLBase::setNumRestarts(const uint32 _numRestarts) {
  this->numRestarts = _numRestarts;
  return true;
}

bool MLBase::setMinChange(const float _minChange) {
  if (minChange < 0) {
    UE_LOG(GRTModule, Warning,
           TEXT(
             "setMinChange(const float minChange) - The minChange must be greater than or equal to 0!"));
    return false;
  }
  this->minChange = _minChange;
  return true;
}

bool MLBase::setLearningRate(const float _learningRate) {
  if (_learningRate > 0) {
    this->learningRate = _learningRate;
    return true;
  }
  return false;
}

bool MLBase::setValidationSetSize(const uint32 _validationSetSize) {
  if ((_validationSetSize > 0) && (_validationSetSize < 100)) {
    this->validationSetSize = _validationSetSize;
    return true;
  }

  UE_LOG(GRTModule, Warning,
         TEXT(
           "setValidationSetSize(const uint32 validationSetSize) - The validation size must be in the range [1 99]!"));
  return false;
}

bool MLBase::setUseValidationSet(const bool _useValidationSet) {
  this->useValidationSet = _useValidationSet;
  return true;
}

bool MLBase::setRandomiseTrainingOrder(const bool _randomiseTrainingOrder) {
  this->randomiseTrainingOrder = _randomiseTrainingOrder;
  return true;
}

bool MLBase::registerTrainingResultsObserver(Observer<TrainingResult>& observer) {
  return trainingResultsObserverManager.registerObserver(observer);
}

bool MLBase::registerTestResultsObserver(Observer<TestInstanceResult>& observer) {
  return testResultsObserverManager.registerObserver(observer);
}

bool MLBase::removeTrainingResultsObserver(
  const Observer<TrainingResult>& observer) {
  return trainingResultsObserverManager.removeObserver(observer);
}

bool MLBase::removeTestResultsObserver(
  const Observer<TestInstanceResult>& observer) {
  return testResultsObserverManager.removeObserver(observer);
}

bool MLBase::removeAllTrainingObservers() {
  return trainingResultsObserverManager.removeAllObservers();
}

bool MLBase::removeAllTestObservers() {
  return testResultsObserverManager.removeAllObservers();
}

bool MLBase::notifyTrainingResultsObservers(const TrainingResult& data) {
  return trainingResultsObserverManager.notifyObservers(data);
}

bool MLBase::notifyTestResultsObservers(const TestInstanceResult& data) {
  return testResultsObserverManager.notifyObservers(data);
}

MLBase * MLBase::getMLBasePointer() {
  return this;
}

const MLBase * MLBase::getMLBasePointer() const {
  return this;
}

Vector<TrainingResult>MLBase::getTrainingResults() const {
  return trainingResults;
}

bool MLBase::saveBaseSettingsToFile(std::fstream& file) const {
  if (!file.is_open()) {
    UE_LOG(GRTModule, Error,
           TEXT("saveBaseSettingsToFile(fstream &file) - The file is not open!"));
    return false;
  }

  file << "Trained: " << trained << std::endl;
  file << "UseScaling: " << useScaling << std::endl;
  file << "NumInputDimensions: " << numInputDimensions << std::endl;
  file << "NumOutputDimensions: " << numOutputDimensions << std::endl;
  file << "NumTrainingIterationsToConverge: " <<
    numTrainingIterationsToConverge << std::endl;
  file << "MinNumEpochs: " << minNumEpochs << std::endl;
  file << "MaxNumEpochs: " << maxNumEpochs << std::endl;
  file << "ValidationSetSize: " << validationSetSize << std::endl;
  file << "LearningRate: " << learningRate << std::endl;
  file << "MinChange: " << minChange << std::endl;
  file << "UseValidationSet: " << useValidationSet << std::endl;
  file << "RandomiseTrainingOrder: " << randomiseTrainingOrder << std::endl;

  return true;
}

bool MLBase::loadBaseSettingsFromFile(std::fstream& file) {
  // Clear any previous setup
  clear();

  if (!file.is_open()) {
    UE_LOG(GRTModule, Error,
           TEXT("loadBaseSettingsFromFile(fstream &file) - The file is not open!"));
    return false;
  }

  std::string word;

  // Load the trained state
  file >> word;

  if (word != "Trained:") {
    UE_LOG(GRTModule, Error,
           TEXT(
             "loadBaseSettingsFromFile(fstream &file) - Failed to read Trained header!"));
    return false;
  }
  file >> trained;

  // Load the scaling state
  file >> word;

  if (word != "UseScaling:") {
    UE_LOG(GRTModule, Error,
           TEXT(
             "loadBaseSettingsFromFile(fstream &file) - Failed to read UseScaling header!"));
    return false;
  }
  file >> useScaling;

  // Load the NumInputDimensions
  file >> word;

  if (word != "NumInputDimensions:") {
    UE_LOG(GRTModule, Error,
           TEXT(
             "loadBaseSettingsFromFile(fstream &file) - Failed to read NumInputDimensions header!"));
    return false;
  }
  file >> numInputDimensions;

  // Load the NumOutputDimensions
  file >> word;

  if (word != "NumOutputDimensions:") {
    UE_LOG(GRTModule, Error,
           TEXT(
             "loadBaseSettingsFromFile(fstream &file) - Failed to read NumOutputDimensions header!"));
    return false;
  }
  file >> numOutputDimensions;

  // Load the numTrainingIterationsToConverge
  file >> word;

  if (word != "NumTrainingIterationsToConverge:") {
    UE_LOG(GRTModule, Error,
           TEXT(
             "loadBaseSettingsFromFile(fstream &file) - Failed to read NumTrainingIterationsToConverge header!"));
    return false;
  }
  file >> numTrainingIterationsToConverge;

  // Load the MinNumEpochs
  file >> word;

  if (word != "MinNumEpochs:") {
    UE_LOG(GRTModule, Error,
           TEXT(
             "loadBaseSettingsFromFile(fstream &file) - Failed to read MinNumEpochs header!"));
    return false;
  }
  file >> minNumEpochs;

  // Load the maxNumEpochs
  file >> word;

  if (word != "MaxNumEpochs:") {
    UE_LOG(GRTModule, Error,
           TEXT(
             "loadBaseSettingsFromFile(fstream &file) - Failed to read MaxNumEpochs header!"));
    return false;
  }
  file >> maxNumEpochs;

  // Load the ValidationSetSize
  file >> word;

  if (word != "ValidationSetSize:") {
    UE_LOG(GRTModule, Error,
           TEXT(
             "loadBaseSettingsFromFile(fstream &file) - Failed to read ValidationSetSize header!"));
    return false;
  }
  file >> validationSetSize;

  // Load the LearningRate
  file >> word;

  if (word != "LearningRate:") {
    UE_LOG(GRTModule, Error,
           TEXT(
             "loadBaseSettingsFromFile(fstream &file) - Failed to read LearningRate header!"));
    return false;
  }
  file >> learningRate;

  // Load the MinChange
  file >> word;

  if (word != "MinChange:") {
    UE_LOG(GRTModule, Error,
           TEXT(
             "loadBaseSettingsFromFile(fstream &file) - Failed to read MinChange header!"));
    return false;
  }
  file >> minChange;

  // Load the UseValidationSet
  file >> word;

  if (word != "UseValidationSet:") {
    UE_LOG(GRTModule, Error,
           TEXT(
             "loadBaseSettingsFromFile(fstream &file) - Failed to read UseValidationSet header!"));
    return false;
  }
  file >> useValidationSet;

  // Load the RandomiseTrainingOrder
  file >> word;

  if (word != "RandomiseTrainingOrder:") {
    UE_LOG(GRTModule, Error,
           TEXT(
             "loadBaseSettingsFromFile(fstream &file) - Failed to read RandomiseTrainingOrder header!"));
    return false;
  }
  file >> randomiseTrainingOrder;
  return true;
}
}
