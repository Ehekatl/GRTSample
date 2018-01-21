#pragma once

#include "../GRT.h"
#include "MLBase.h"
#include <map>

#ifndef GRT_CLASSIFIER_HEADER
# define GRT_CLASSIFIER_HEADER
# define DEFAULT_NULL_LIKELIHOOD_VALUE 0
# define DEFAULT_NULL_DISTANCE_VALUE 0
#endif // GRT_CLASSIFIER_HEADER

namespace GRT {
/**
   @brief This is the main base class that all GRT Classification algorithms
      should inherit from.

   A large number of the functions in this class are virtual and simply return
      false as these functions must be overridden by the inheriting class.
 */
class GRT_API Classifier : public MLBase {
public:

  enum ClassifierModes { STANDARD_CLASSIFIER_MODE = 0,
                         TIMESERIES_CLASSIFIER_MODE }; ///<An enum that
                                                       // indicates if the
                                                       // classifier supports
                                                       // timeseries data

  /**
     Default Classifier Constructor
     @param classifierId: the id of the parent classifier (e.g., DecisionTree)
   */
  Classifier(const FString& classifierId = "");

  /**
     Default Classifier Destructor
   */
  virtual ~Classifier(void);

  /**
     This is the base deep copy function for the Classifier modules. This
        function should be overwritten by the derived class.
     This deep copies the variables and models from the classifier pointer to
        this classifier instance.

     @param classifier: a pointer to the Classifier base class, this should be
        pointing to another instance of a matching derived class
     @return returns true if the clone was successfully, false otherwise (the
        Classifier base class will always return false)
   */
  virtual bool deepCopyFrom(const Classifier *classifier) {
    return false;
  }

  /**
     This copies the Classifier base class variables from the classifier pointer
        to this instance.

     @param classifier: a pointer to a classifier from which the values will be
        copied to this instance
     @return returns true if the copy was successfully, false otherwise
   */
  bool           copyBaseVariables(const Classifier *classifier);

  /**
     This resets the classifier.
     This overrides the reset function in the MLBase base class.

     @return returns true if the classifier was reset, false otherwise
   */
  virtual bool   reset();

  /**
     This function clears the classifier module, removing any trained model and
        setting all the base variables to their default values.

     @return returns true if the derived class was cleared successfully, false
        otherwise
   */
  virtual bool   clear();

  /**
     Returns the classifier type as a string.

     @return returns the classifier type as a string
   */
  FString        getClassifierType() const;

  /**
     Returns true if the classifier instance supports null rejection, false
        otherwise.

     @return returns true if the classifier instance supports null rejection,
        false otherwise
   */
  bool           getSupportsNullRejection() const;

  /**
     Returns true if nullRejection is enabled.

     @return returns true if nullRejection is enabled, false otherwise
   */
  bool           getNullRejectionEnabled() const;

  /**
     Returns the current nullRejectionCoeff value.
     The nullRejectionCoeff parameter is a multipler controlling the null
        rejection threshold for each class.

     @return returns the current nullRejectionCoeff value
   */
  float          getNullRejectionCoeff() const;

  /**
     Returns the current maximumLikelihood value.
     The maximumLikelihood value is computed during the prediction phase and is
        the likelihood of the most likely model.
     This value will return 0 if a prediction has not been made.

     @return returns the current maximumLikelihood value
   */
  float          getMaximumLikelihood() const;

  /**
     Returns the current bestDistance value.
     The bestDistance value is computed during the prediction phase and is
        either the minimum or maximum distance, depending on the algorithm.
     This value will return 0 if a prediction has not been made.

     @return returns the current bestDistance value
   */
  float          getBestDistance() const;

  /**
     This function returns the estimated gesture phase from the most recent
        prediction.  This value is only relevant if the classifier supports
        timeseries classification.

     @return float representing the gesture phase value from the most likely
        class from the most recent prediction
   */
  float          getPhase() const;

  /**
     This function returns the estimated training set accuracy from the most
        recent round of training.  This value is only relevant if the classifier
        has been trained.

     @return float representing the training set accuracy
   */
  float          getTrainingSetAccuracy() const;

  /**
     Gets the number of classes in trained model.

     @return returns the number of classes in the trained model, a value of 0
        will be returned if the model has not been trained
   */
  virtual uint32 getNumClasses() const;

  /**
     Gets the index of the query classLabel in the classLabels Vector. If the
        query classLabel does not exist in the classLabels Vector
     then the function will return zero.

     @param classLabel: the query classLabel
     @return returns index of the query classLabel in the classLabels Vector
   */
  uint32         getClassLabelIndexValue(const uint32 classLabel) const;

  /**
     Gets the predicted class label from the last prediction.

     @return returns the label of the last predicted class, a value of 0 will be
        returned if the model has not been trained
   */
  uint32         getPredictedClassLabel() const;

  /**
     Gets a Vector of the class likelihoods from the last prediction, this will
        be an N-dimensional Vector, where N is the number of classes in the
        model.
     The exact form of these likelihoods depends on the classification
        algorithm.

     @return returns a Vector of the class likelihoods from the last prediction,
        an empty Vector will be returned if the model has not been trained
   */
  VectorFloat    getClassLikelihoods() const;

  /**
     Gets a Vector of the class distances from the last prediction, this will be
        an N-dimensional Vector, where N is the number of classes in the model.
     The exact form of these distances depends on the classification algorithm.

     @return returns a Vector of the class distances from the last prediction,
        an empty Vector will be returned if the model has not been trained
   */
  VectorFloat    getClassDistances() const;

  /**
     Gets a Vector containing the null rejection thresholds for each class, this
        will be an N-dimensional Vector, where N is the number of classes in the
        model.

     @return returns a Vector containing the null rejection thresholds for each
        class, an empty Vector will be returned if the model has not been
        trained
   */
  VectorFloat    getNullRejectionThresholds() const;

  /**
     Gets a Vector containing the label each class represents, this will be an
        N-dimensional Vector, where N is the number of classes in the model.
     This is useful if the model was trained with non-monotonically class labels
        (i.e. class labels such as [1, 3, 6, 9, 12] instead of [1, 2, 3, 4, 5]).

     @return returns a Vector containing the class labels for each class, an
        empty Vector will be returned if the model has not been trained
   */
  Vector<uint32> getClassLabels() const;

  /**
     Gets a Vector of the ranges used to scale the data for training and
        prediction, these ranges are only used if the classifier has been
        trained
     with the #useScaling flag set to true. This should be an N-dimensional
        Vector, where N is the number of features in your data.

     @return returns a Vector containing the ranges used to scale the data for
        classification, an empty Vector will be returned if the model has not
        been trained
   */
  Vector<MinMax> getRanges() const;

  /**
     Sets if the classifier should use nullRejection.

     If set to true then the classifier will reject a predicted class label if
        the likelihood of the prediction is below (or above depending on the
     algorithm) the models rejectionThreshold. If a prediction is rejected then
        the default null class label of 0 will be returned.
     If set to false then the classifier will simply return the most likely
        predicted class.

     @param useNullRejection: the new null rejection state
     @return returns true if nullRejection was updated successfully, false
        otherwise
   */
  bool           enableNullRejection(const bool useNullRejection);

  /**
     Sets the nullRejectionCoeff, this is a multiplier controlling the null
        rejection threshold for each class.

     @param nullRejectionCoeff: the new null rejection value
     @return returns true if nullRejectionCoeff was updated successfully, false
        otherwise
   */
  virtual bool   setNullRejectionCoeff(const float nullRejectionCoeff);

  /**
     Manually sets the nullRejectionThresholds, these are the thresholds used
        for null rejection for each class.
     This needs to be called after the model has been trained. Calling the
     #setNullRejectionCoeff or #recomputeNullRejectionThresholds
     functions will override these values. The size of the
        newRejectionThresholds Vector must match the number of classes in the
        model.

     @param newRejectionThresholds: the new rejection thresholds
     @return returns true if nullRejectionThresholds were updated successfully,
        false otherwise
   */
  virtual bool   setNullRejectionThresholds(
    const VectorFloat& newRejectionThresholds);

  /**
     Recomputes the null rejection thresholds for each model.

     @return returns true if the nullRejectionThresholds were updated
        successfully, false otherwise
   */
  virtual bool recomputeNullRejectionThresholds() {
    return false;
  }

  /**
     Indicates if the classifier can be used to classify timeseries data.
     If true then the classifier can accept training data in the
        LabelledTimeSeriesClassificationData format.

     return returns true if the classifier can be used to classify timeseries
        data, false otherwise
   */
  bool getTimeseriesCompatible() const {
    return classifierMode == TIMESERIES_CLASSIFIER_MODE;
  }

  /**
     Defines a map between a string (which will contain the name of the
        classifier, such as ANBC) and a function returns a new instance of that
        classifier
   */
  typedef std::map<FString, Classifier *(*)()>StringClassifierMap;

  /**
     Creates a new classifier instance based on the input string (which should
        contain the name of a valid classifier such as KNN).

     @param id: the name of the classifier
     @return Classifier*: a pointer to the new instance of the classifier
   */
  static Classifier   * create(const FString& id);

  /**
     Creates a new classifier instance based on the current classifierType
        string value.

     @return Classifier*: a pointer to the new instance of the classifier
   */
  Classifier          * create() const;

  /**
     This creates a new Classifier instance and deep copies the variables and
        models from this instance into the deep copy.
     The function will then return a pointer to the new instance. It is up to
        the user who calls this function to delete the dynamic instance
     when they are finished using it.

     @return returns a pointer to a new Classifier instance which is a deep copy
        of this instance
   */
  Classifier          * deepCopy() const;

  /**
     Returns a pointer to the classifier.

     @return returns a pointer the current classifier
   */
  const Classifier    * getClassifierPointer() const;

  /**
     Returns a pointer to this classifier. This is useful for a derived class so
        it can get easy access to this base classifier.

     @return Classifier&: a reference to this classifier
   */
  const Classifier    & getBaseClassifier() const;

  /**
     Returns a Vector of the names of all classifiers that have been registered
        with the base classifier.

     @return Vector< string >: a Vector containing the names of the classifiers
        that have been registered with the base classifier
   */
  static Vector<FString>getRegisteredClassifiers();

protected:

  /**
     Saves the core base settings to a file.

     @return returns true if the base settings were saved, false otherwise
   */
  bool saveBaseSettingsToFile(std::fstream& file) const;

  /**
     Loads the core base settings from a file.

     @return returns true if the base settings were loaded, false otherwise
   */
  bool loadBaseSettingsFromFile(std::fstream& file);

  bool   supportsNullRejection;
  bool   useNullRejection;
  uint32 numClasses;
  uint32 predictedClassLabel;
  uint32 classifierMode;
  float  nullRejectionCoeff;
  float  maxLikelihood;
  float  bestDistance;
  float  phase;
  float  trainingSetAccuracy;
  VectorFloat    classLikelihoods;
  VectorFloat    classDistances;
  VectorFloat    nullRejectionThresholds;
  Vector<uint32> classLabels;
  Vector<MinMax> ranges;

  /**
     This function returns the classifier map, only one map should exist across
        all classifiers.
     If a map has not been created then one will be created, otherwise the
        current map will be returned.
     @return returns a pointer to the classifier map
   */
  static StringClassifierMap* getMap() {
    if (!stringClassifierMap)   stringClassifierMap = new StringClassifierMap;
    return stringClassifierMap;
  }

private:

  static StringClassifierMap *stringClassifierMap;
  static uint32 numClassifierInstances;
};

template<typename T>
Classifier* createNewClassifierInstance() {
  return new T;
} ///< Returns a pointer to a new instance of the template class, the caller is

// responsible for deleting the pointer

/**
   @brief This class provides an interface for classes to register themselves
      with the classifier base class, this enables Classifier algorithms to
   be automatically be created from just a string, e.g.: Classifier *knn =
      create( "KNN" );
 */
template<typename T>
class RegisterClassifierModule : public Classifier {
public:

  RegisterClassifierModule(FString const& newModuleId) {
    getMap()->insert(std::pair<FString, Classifier *(*)()>(newModuleId,
                                                           &
                                                           createNewClassifierInstance
                                                           <T>));
  }
};
}
