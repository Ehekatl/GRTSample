#include "../GRT.h"
#include "Classifier.h"

namespace GRT {
Classifier::StringClassifierMap *Classifier::stringClassifierMap = NULL;
uint32 Classifier::numClassifierInstances                        = 0;

Classifier * Classifier::create(const FString& id) {
  // This function maps the input string and returns a pointer to a new instance

  StringClassifierMap::iterator iter = getMap()->find(id);

  if (iter == getMap()->end()) {
    // If the iterator points to the end of the map, then no match was found so
    // return NULL
    return NULL;
  }

  return iter->second();
}

Classifier * Classifier::create() const {
  return create(MLBase::getId());
}

Classifier * Classifier::deepCopy() const {
  Classifier *newInstance = create(MLBase::getId());

  if (newInstance == NULL) return NULL;

  if (!newInstance->deepCopyFrom(this)) {
    delete newInstance;
    return NULL;
  }
  return newInstance;
}

const Classifier * Classifier::getClassifierPointer() const {
  return this;
}

Vector<FString>Classifier::getRegisteredClassifiers() {
  Vector<FString> registeredClassifiers;

  StringClassifierMap::iterator iter = getMap()->begin();

  while (iter != getMap()->end()) {
    registeredClassifiers.push_back(iter->first);
    ++iter; // ++iter is faster than iter++ as it does not require a copy/move
            // operator
  }
  return registeredClassifiers;
}

Classifier::Classifier(const FString& id) : MLBase(id, MLBase::CLASSIFIER)
{
  classifierMode        = STANDARD_CLASSIFIER_MODE;
  supportsNullRejection = false;
  useNullRejection      = false;
  numInputDimensions    = 0;
  numOutputDimensions   = 0;
  numClasses            = 0;
  predictedClassLabel   = 0;
  maxLikelihood         = 0;
  bestDistance          = 0;
  phase                 = 0;
  trainingSetAccuracy   = 0;
  nullRejectionCoeff    = 5;
  numClassifierInstances++;
}

Classifier::~Classifier(void) {
  if (--numClassifierInstances == 0) {
    delete stringClassifierMap;
    stringClassifierMap = NULL;
  }
}

bool Classifier::copyBaseVariables(const Classifier *classifier) {
  if (classifier == NULL) {
    UE_LOG(GRTModule, Error,
           TEXT(
             "copyBaseVariables(const Classifier *classifier) - Classifier is NULL!"));
    return false;
  }

  if (!this->copyMLBaseVariables(classifier)) {
    return false;
  }

  this->classifierMode          = classifier->classifierMode;
  this->supportsNullRejection   = classifier->supportsNullRejection;
  this->useNullRejection        = classifier->useNullRejection;
  this->numClasses              = classifier->numClasses;
  this->predictedClassLabel     = classifier->predictedClassLabel;
  this->classifierMode          = classifier->classifierMode;
  this->nullRejectionCoeff      = classifier->nullRejectionCoeff;
  this->maxLikelihood           = classifier->maxLikelihood;
  this->bestDistance            = classifier->bestDistance;
  this->phase                   = classifier->phase;
  this->trainingSetAccuracy     = classifier->trainingSetAccuracy;
  this->classLabels             = classifier->classLabels;
  this->classLikelihoods        = classifier->classLikelihoods;
  this->classDistances          = classifier->classDistances;
  this->nullRejectionThresholds = classifier->nullRejectionThresholds;
  this->ranges                  = classifier->ranges;

  return true;
}

bool Classifier::reset() {
  // Reset the base class
  MLBase::reset();

  // Reset the classifier
  predictedClassLabel = GRT_DEFAULT_NULL_CLASS_LABEL;
  maxLikelihood       = 0;
  bestDistance        = 0;
  phase               = 0;

  if (trained) {
    classLikelihoods.clear();
    classDistances.clear();
    classLikelihoods.resize(numClasses, 0);
    classDistances.resize(numClasses, 0);
  }
  return true;
}

bool Classifier::clear() {
  // Clear the MLBase variables
  MLBase::clear();

  // Clear the classifier variables
  predictedClassLabel = GRT_DEFAULT_NULL_CLASS_LABEL;
  maxLikelihood       = 0;
  bestDistance        = 0;
  phase               = 0;
  trainingSetAccuracy = 0;
  classLikelihoods.clear();
  classDistances.clear();
  nullRejectionThresholds.clear();
  classLabels.clear();
  ranges.clear();

  return true;
}

FString Classifier::getClassifierType() const {
  return MLBase::getId();
}

bool Classifier::getSupportsNullRejection() const {
  return supportsNullRejection;
}

bool Classifier::getNullRejectionEnabled() const {
  return useNullRejection;
}

float Classifier::getNullRejectionCoeff() const {
  return nullRejectionCoeff;
}

float Classifier::getMaximumLikelihood() const {
  if (trained) return maxLikelihood;

  return DEFAULT_NULL_LIKELIHOOD_VALUE;
}

float Classifier::getPhase() const {
  return phase;
}

float Classifier::getTrainingSetAccuracy() const {
  return trainingSetAccuracy;
}

float Classifier::getBestDistance() const {
  if (trained) return bestDistance;

  return DEFAULT_NULL_DISTANCE_VALUE;
}

uint32 Classifier::getNumClasses() const {
  return numClasses;
}

uint32 Classifier::getClassLabelIndexValue(const uint32 classLabel) const {
  for (uint32 i = 0; i < classLabels.size(); i++) {
    if (classLabel == classLabels[i]) return i;
  }
  return 0;
}

uint32 Classifier::getPredictedClassLabel() const {
  if (trained) return predictedClassLabel;

  return 0;
}

VectorFloat Classifier::getClassLikelihoods() const {
  if (trained) return classLikelihoods;

  return VectorFloat();
}

VectorFloat Classifier::getClassDistances() const {
  if (trained) return classDistances;

  return VectorFloat();
}

VectorFloat Classifier::getNullRejectionThresholds() const {
  if (trained) return nullRejectionThresholds;

  return VectorFloat();
}

Vector<uint32>Classifier::getClassLabels() const {
  return classLabels;
}

Vector<MinMax>Classifier::getRanges() const {
  return ranges;
}

bool Classifier::enableNullRejection(const bool _useNullRejection) {
  this->useNullRejection = _useNullRejection;
  return true;
}

bool Classifier::setNullRejectionCoeff(const float _nullRejectionCoeff) {
  if (nullRejectionCoeff > 0) {
    this->nullRejectionCoeff = _nullRejectionCoeff;
    return true;
  }
  return false;
}

bool Classifier::setNullRejectionThresholds(
  const VectorFloat& newRejectionThresholds) {
  if (newRejectionThresholds.getSize() == getNumClasses()) {
    nullRejectionThresholds = newRejectionThresholds;
    return true;
  }
  return false;
}

const Classifier& Classifier::getBaseClassifier() const {
  return *this;
}

bool Classifier::saveBaseSettingsToFile(std::fstream& file) const {
  if (!file.is_open()) {
    UE_LOG(GRTModule, Error,
           TEXT("saveBaseSettingsToFile(fstream &file) - The file is not open!"));
    return false;
  }

  if (!MLBase::saveBaseSettingsToFile(file)) return false;

  file << "UseNullRejection: " << useNullRejection << std::endl;
  file << "ClassifierMode: " << classifierMode << std::endl;
  file << "NullRejectionCoeff: " << nullRejectionCoeff << std::endl;

  if (trained) {
    file << "NumClasses: " << numClasses << std::endl;

    file << "NullRejectionThresholds: ";

    if (useNullRejection && nullRejectionThresholds.size()) {
      for (uint32 i = 0; i < nullRejectionThresholds.size(); i++) {
        file << " " << nullRejectionThresholds[i];
      }
      file << std::endl;
    }
    else {
      for (uint32 i = 0; i < numClasses; i++) {
        file << " " << 0.0;
      }
      file << std::endl;
    }

    file << "ClassLabels: ";

    for (uint32 i = 0; i < classLabels.size(); i++) {
      file << " " << classLabels[i];
    }
    file << std::endl;

    if (useScaling) {
      file << "Ranges: " << std::endl;

      for (uint32 i = 0; i < ranges.size(); i++) {
        file << ranges[i].minValue << "\t" << ranges[i].maxValue << std::endl;
      }
    }
  }
  return true;
}

bool Classifier::loadBaseSettingsFromFile(std::fstream& file) {
  if (!file.is_open()) {
    UE_LOG(GRTModule, Error,
           TEXT("loadBaseSettingsFromFile(fstream &file) - The file is not open!"));
    return false;
  }

  // Try and load the base settings from the file
  if (!MLBase::loadBaseSettingsFromFile(file)) {
    return false;
  }

  std::string word;

  // Load if the number of clusters
  file >> word;

  if (word != "UseNullRejection:") {
    UE_LOG(GRTModule, Error,
           TEXT(
             "loadBaseSettingsFromFile(fstream &file) - Failed to read UseNullRejection header!"));
    clear();
    return false;
  }
  file >> useNullRejection;

  // Load if the classifier mode
  file >> word;

  if (word != "ClassifierMode:") {
    UE_LOG(GRTModule, Error,
           TEXT(
             "loadBaseSettingsFromFile(fstream &file) - Failed to read ClassifierMode header!"));
    clear();
    return false;
  }
  file >> classifierMode;

  // Load if the null rejection coeff
  file >> word;

  if (word != "NullRejectionCoeff:") {
    UE_LOG(GRTModule, Error,
           TEXT(
             "loadBaseSettingsFromFile(fstream &file) - Failed to read NullRejectionCoeff header!"));
    clear();
    return false;
  }
  file >> nullRejectionCoeff;

  // If the model is trained then load the model settings
  if (trained) {
    // Load the number of classes
    file >> word;

    if (word != "NumClasses:") {
      UE_LOG(GRTModule, Error,
             TEXT(
               "loadBaseSettingsFromFile(fstream &file) - Failed to read NumClasses header!"));
      clear();
      return false;
    }
    file >> numClasses;

    // Load the null rejection thresholds
    file >> word;

    if (word != "NullRejectionThresholds:") {
      UE_LOG(GRTModule, Error,
             TEXT(
               "loadBaseSettingsFromFile(fstream &file) - Failed to read NullRejectionThresholds header!"));
      clear();
      return false;
    }
    nullRejectionThresholds.resize(numClasses);

    for (uint32 i = 0; i < nullRejectionThresholds.size(); i++) {
      file >> nullRejectionThresholds[i];
    }

    // Load the class labels
    file >> word;

    if (word != "ClassLabels:") {
      UE_LOG(GRTModule, Error,
             TEXT(
               "loadBaseSettingsFromFile(fstream &file) - Failed to read ClassLabels header!"));
      clear();
      return false;
    }
    classLabels.resize(numClasses);

    for (uint32 i = 0; i < classLabels.size(); i++) {
      file >> classLabels[i];
    }

    if (useScaling) {
      // Load if the Ranges
      file >> word;

      if (word != "Ranges:") {
        UE_LOG(GRTModule, Error,
               TEXT(
                 "loadBaseSettingsFromFile(fstream &file) - Failed to read Ranges header!"));
        clear();
        return false;
      }
      ranges.resize(numInputDimensions);

      for (uint32 i = 0; i < ranges.size(); i++) {
        file >> ranges[i].minValue;
        file >> ranges[i].maxValue;
      }
    }
  }
  return true;
}
}
