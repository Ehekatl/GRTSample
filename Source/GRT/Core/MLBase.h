#pragma once

#include "../GRT.h"
#include "GRTBase.h"
#include "../Types/VectorFloat.h" // Forwarded def for TestInstanceResult
#include "../Utility/ObserverManager.h"
#include "../Utility/TrainingResult.h"
#include "../Utility/TestInstanceResult.h"
#include "../Utility/DataType.h"
#include "../Types/TimeSeriesClassificationData.h"

#ifndef GRT_MLBASE_HEADER
# define GRT_MLBASE_HEADER
# define DEFAULT_NULL_LIKELIHOOD_VALUE 0
# define DEFAULT_NULL_DISTANCE_VALUE 0
#endif // GRT_MLBASE_HEADER

namespace GRT {
/**
   Define the class for handling TrainingResult callbacks
 */
class GRT_API TrainingResultsObserverManager : public ObserverManager<
                                                 TrainingResult>{
public:

  TrainingResultsObserverManager() {}

  virtual ~TrainingResultsObserverManager() {}
};

/**
   Define the class for handling TestInstanceResult callbacks
 */
class GRT_API TestResultsObserverManager : public ObserverManager<
                                             TestInstanceResult>{
public:

  TestResultsObserverManager() {}

  virtual ~TestResultsObserverManager() {}
};

/**
   @brief This is the main base class that all GRT machine learning algorithms
      should inherit from.

   A large number of the functions in this class are virtual and simply return
      false as these functions must be overridden by the inheriting class.
 */
class GRT_API MLBase : public GRTBase, public Observer<TrainingResult>,
                       public Observer<TestInstanceResult>{
public:

  enum BaseType { BASE_TYPE_NOT_SET = 0, CLASSIFIER, REGRESSIFIER, CLUSTERER,
                  PRE_PROCSSING, POST_PROCESSING, FEATURE_EXTRACTION, CONTEXT }; ///<Enum
                                                                                 //
                                                                                 //
                                                                                 // that
                                                                                 //
                                                                                 //
                                                                                 // defines
                                                                                 //
                                                                                 //
                                                                                 // the
                                                                                 //
                                                                                 //
                                                                                 // type
                                                                                 //
                                                                                 //
                                                                                 // of
                                                                                 //
                                                                                 //
                                                                                 // inherited
                                                                                 //
                                                                                 //
                                                                                 // class

  /**
     Default MLBase Constructor
     @param id: the id of the inheriting class
     @param type: the type of the inheriting class (e.g., classifier,
        regressifier, etc.)
   */
  MLBase(const FString& id = "",
         const BaseType type = BASE_TYPE_NOT_SET);

  /**
     Default MLBase Destructor
   */
  virtual ~MLBase(void);

  /**
     This copies all the MLBase variables from the instance mlBaseA to the
        instance mlBaseA.

     @param mlBase: a pointer to a MLBase class from which the values will be
        copied to the instance that calls the function
     @return returns true if the copy was successfully, false otherwise
   */
  bool            copyMLBaseVariables(const MLBase *mlBase);

  /**
     This is the main training interface for TimeSeriesClassificationData.
     By default it will call the train_ function, unless it is overwritten by
        the derived class.

     @param trainingData: the training data that will be used to train the ML
        model
     @return returns true if the classifier was successfully trained, false
        otherwise
   */
  virtual bool    train(TimeSeriesClassificationData trainingData);

  /**
     This is the main training interface for referenced
        TimeSeriesClassificationData. This should be overwritten by the derived
        class.

     @param trainingData: a reference to the training data that will be used to
        train the ML model
     @return returns true if the classifier was successfully trained, false
        otherwise
   */
  virtual bool    train_(TimeSeriesClassificationData& trainingData);

  /**
     This is the main training interface for MatrixFloat data.
     By default it will call the train_ function, unless it is overwritten by
        the derived class.

     @param trainingData: the training data that will be used to train the ML
        model
     @return returns true if the classifier was successfully trained, false
        otherwise
   */
  virtual bool    train(MatrixFloat data);

  /**
     This is the main training interface for referenced MatrixFloat data. This
        should be overwritten by the derived class.

     @param trainingData: a reference to the training data that will be used to
        train the ML model
     @return returns true if the classifier was successfully trained, false
        otherwise
   */
  virtual bool    train_(MatrixFloat& data);

  /**
     This is the main prediction interface for all the GRT machine learning
        algorithms.
     By default it will call the predict_ function, unless it is overwritten by
        the derived class.

     @param inputVector: the new input vector for prediction
     @return returns true if the prediction was completed successfully, false
        otherwise (the base class always returns false)
   */
  virtual bool    predict(VectorFloat inputVector);

  /**
     This is the main prediction interface for all the GRT machine learning
        algorithms. This should be overwritten by the derived class.

     @param inputVector: a reference to the input vector for prediction
     @return returns true if the prediction was completed successfully, false
        otherwise (the base class always returns false)
   */
  virtual bool    predict_(VectorFloat& inputVector);

  /**
     This is the prediction interface for time series data.
     By default it will call the predict_ function, unless it is overwritten by
        the derived class.

     @param inputMatrix: the new input matrix for prediction
     @return returns true if the prediction was completed successfully, false
        otherwise (the base class always returns false)
   */
  virtual bool    predict(MatrixFloat inputMatrix);

  /**
     This is the prediction interface for time series data. This should be
        overwritten by the derived class.

     @param inputMatrix: a reference to the new input matrix for prediction
     @return returns true if the prediction was completed successfully, false
        otherwise (the base class always returns false)
   */
  virtual bool    predict_(MatrixFloat& inputMatrix);

  /**
     This is the main mapping interface for all the GRT machine learning
        algorithms.
     By default it will call the map_ function, unless it is overwritten by the
        derived class.

     @param inputVector: the input vector for mapping/regression
     @return returns true if the mapping was completed successfully, false
        otherwise (the base class always returns false)
   */
  virtual bool    map(VectorFloat inputVector);

  /**
     This is the main mapping interface by reference for all the GRT machine
        learning algorithms. This should be overwritten by the derived class.

     @param inputVector: a reference to the input vector for mapping/regression
     @return returns true if the mapping was completed successfully, false
        otherwise (the base class always returns false)
   */
  virtual bool    map_(VectorFloat& inputVector);

  /**
     This is the main reset interface for all the GRT machine learning
        algorithms.
     It should be used to reset the model (i.e. set all values back to default
        settings). If you want to completely clear the model
     (i.e. clear any learned weights or values) then you should use the clear
        function.

     @return returns true if the derived class was reset successfully, false
        otherwise (the base class always returns true)
   */
  virtual bool    reset();

  /**
     This is the main clear interface for all the GRT machine learning
        algorithms.
     It will completely clear the ML module, removing any trained model and
        setting all the base variables to their default values.

     @return returns true if the derived class was cleared successfully, false
        otherwise
   */
  virtual bool    clear();

  /**
     This saves the model to a file.

     @param filename: the name of the file to save the model to
     @return returns true if the model was saved successfully, false otherwise
   */
  virtual bool    save(const FString& filename) const;

  /**
     This saves the model to a file.

     @param filename: the name of the file to save the model to
     @return returns true if the model was saved successfully, false otherwise
   */
  virtual bool    load(const FString& filename);

  /**
     This saves the trained model to a file.
     This function should be overwritten by the derived class.

     @param file: a reference to the file the model will be saved to
     @return returns true if the model was saved successfully, false otherwise
   */
  virtual bool    save(std::fstream& file) const;

  /**
     This loads a trained model from a file.
     This function should be overwritten by the derived class.

     @param file: a reference to the file the model will be loaded from
     @return returns true if the model was loaded successfully, false otherwise
   */
  virtual bool    load(std::fstream& file);

  /**
     This function adds the current model to the formatted stream.
     This function should be overwritten by the derived class.

     @param file: a reference to the stream the model will be added to
     @return returns true if the model was added successfully, false otherwise
   */
  virtual bool    getModel(std::ostream& stream) const;

  /**
     Gets the current model and settings as a FString.

     @return returns a FString containing the model
   */
  virtual FString getModelAsString() const;

  /**
     Gets the expected input data type for the module

     @return returns the expected input data type
   */
  DataType        getInputType() const;

  /**
     Gets the expected output data type for the module

     @return returns the expected output data type
   */
  DataType        getOutputType() const;

  /**
     Gets the current ML base type.

     @return returns an enum representing the current ML base type, this will be
        one of the BaseType enumerations
   */
  BaseType        getType() const;

  /**
     Gets the number of input dimensions in trained model.

     @return returns the number of input dimensions
   */
  uint32          getNumInputDimensions() const;

  /**
     Gets the number of output dimensions in trained model.

     @return returns the number of output dimensions
   */
  uint32          getNumOutputDimensions() const;

  /**
     Gets the minimum number of epochs. This is the minimum number of epochs
        that can elapse with no change between two training epochs.
     An epoch is a complete iteration of all training samples.

     @return returns the minimum number of epochs
   */
  uint32          getMinNumEpochs() const;

  /**
     Gets the maximum number of epochs. This value controls the maximum number
        of epochs that can be used by the training algorithm.
     An epoch is a complete iteration of all training samples.

     @return returns the maximum number of epochs
   */
  uint32          getMaxNumEpochs() const;

  /**
     Gets the batch size. This value controls the number of samples that can be
        used by the training algorithm.

     @return returns the batch size
   */
  uint32          getBatchSize() const;

  /**
     Gets the number of times a learning algorithm can restart during training.
     @return returns the number of times a learning algorithm can restart during
        training
   */
  uint32          getNumRestarts() const;

  /**
     Gets the size (as a percentage) of the validation set (if one should be
        used). If this value returned 20 this would mean that
     20% of the training data would be set aside to create a validation set and
        the other 80% would be used to actually train the regression model.
     This will only happen if the useValidationSet parameter is set to true,
        otherwise 100% of the training data will be used to train the regression
        model.

     @return returns the size of the validation set
   */
  uint32          getValidationSetSize() const;

  /**
     Gets the number of training iterations that were required for the algorithm
        to converge.

     @return returns the number of training iterations required for the training
        algorithm to converge, a value of 0 will be returned if the model has
        not been trained
   */
  uint32          getNumTrainingIterationsToConverge() const;

  /**
     Gets the current learningRate value, this is value used to update the
        weights at each step of a learning algorithm such as stochastic gradient
        descent.

     @return returns the current learningRate value
   */
  float           getLearningRate() const;

  /**
     Gets the root mean squared error on the training data during the training
        phase.

     @return returns the RMS error (on the training data during the training
        phase)
   */
  float           getRMSTrainingError() const;

  /**
     Gets the total squared error on the training data during the training
        phase.

     @return returns the total squared error (on the training data during the
        training phase)
   */
  float           getTotalSquaredTrainingError() const;

  /**
     Gets the root mean squared error on the validation data during the training
        phase, this will be zero if no validation set was used.

     @return returns the RMS error (on the validation data during the training
        phase)
   */
  float           getRMSValidationError() const;

  /**
     Gets the accuracy of the validation set on the trained model, only valid if
        the model was trained with useValidationSet=true.

     @return returns the accuracy of validation set on the trained model
   */
  float           getValidationSetAccuracy() const;

  /**
     Gets the precision of the validation set on the trained model, only valid
        if the model was trained with useValidationSet=true.

     @return returns the precision of the validation set on the trained model
   */
  VectorFloat     getValidationSetPrecision() const;

  /**
     Gets the recall of the validation set on the trained model, only valid if
        the model was trained with useValidationSet=true.

     @return returns the recall of the validation set on the trained model
   */
  VectorFloat     getValidationSetRecall() const;

  /**
     Returns true if a validation set should be used for training. If true, then
        the training dataset will be partitioned into a smaller training dataset
     and a validation set.

     The size of the partition is controlled by the validationSetSize parameter,
        for example, if the validationSetSize parameter is 20 then 20% of the
     training data will be used for a validation set leaving 80% of the original
        data to train the model.

     @return returns true if a validation set should be used for training, false
        otherwise
   */
  bool            getUseValidationSet() const;

  /**
     Gets if the model for the derived class has been successfully trained.

     @return returns true if the model for the derived class has been
        successfully trained, false otherwise
   */
  bool            getTrained() const;

  /**
     Returns true if the training algorithm converged during the most recent
        training process.
     This function will return false if the model has not been trained.

     @return returns true if the training algorithm converged successfully,
        false otherwise
   */
  bool            getConverged() const;

  /**
     Gets if the scaling has been enabled.

     @return returns true if scaling is enabled, false otherwise
   */
  bool            getScalingEnabled() const;

  /**
     Gets if the derived class type is CLASSIFIER.

     @return returns true if the derived class type is CLASSIFIER, false
        otherwise
   */
  bool            getIsBaseTypeClassifier() const;

  /**
     Gets if the derived class type is REGRESSIFIER.

     @return returns true if the derived class type is REGRESSIFIER, false
        otherwise
   */
  bool            getIsBaseTypeRegressifier() const;

  /**
     Gets if the derived class type is CLUSTERER.

     @return returns true if the derived class type is CLUSTERER, false
        otherwise
   */
  bool            getIsBaseTypeClusterer() const;

  /**
     Sets if scaling should be used during the training and prediction phases.

     @return returns true the scaling parameter was updated, false otherwise
   */
  bool            enableScaling(const bool useScaling);

  /**
     Sets the maximum number of epochs (a complete iteration of all training
        samples) that can be run during the training phase.
     The maxNumIterations value must be greater than zero.

     @param maxNumIterations: the maximum number of iterations value, must be
        greater than zero
     @return returns true if the value was updated successfully, false otherwise
   */
  bool            setMaxNumEpochs(const uint32 maxNumEpochs);

  /**
     Sets the batch size used during the training phase.

     @param batchSize: the batch size
     @return returns true if the value was updated successfully, false otherwise
   */
  bool            setBatchSize(const uint32 batchSize);

  /**
     Sets the minimum number of epochs (a complete iteration of all training
        samples) that can elapse with no change between two training epochs.

     @param minNumEpochs: the minimum number of epochs that can elapse with no
        change between two training epochs
     @return returns true if the value was updated successfully, false otherwise
   */
  bool            setMinNumEpochs(const uint32 minNumEpochs);

  /**
     Sets the number of times a learning algorithm can restart during training.
        This is used to restart the training for algorithms that can get stuck
        if they
     start with bad random values.

     @param numRestarts: number of times a learning algorithm can restart during
        training
     @return returns true if the value was updated successfully, false otherwise
   */
  bool            setNumRestarts(const uint32 numRestarts);

  /**
     Sets the minimum change that must be achieved between two training epochs
        for the training to continue.
     The minChange value must be greater than zero.

     @param minChange: the minimum change value, must be greater than zero
     @return returns true if the value was updated successfully, false otherwise
   */
  bool            setMinChange(const float minChange);

  /**
     Sets the learningRate. This is used to update the weights at each step of
        learning algorithms such as stochastic gradient descent.
     The learningRate value must be greater than zero.

     @param learningRate: the learningRate value used during the training phase,
        must be greater than zero
     @return returns true if the value was updated successfully, false otherwise
   */
  bool            setLearningRate(const float learningRate);

  /**
     Sets the size of the validation set used by some learning algorithms for
        training. This value represents the percentage of the main
     dataset that will be used for training.  For example, if the
        validationSetSize parameter is 20 then 20% of the training data will be
     used for a validation set leaving 80% of the original data to train the
        model.

     @param validationSetSize: the new validation set size (as a percentage)
     @return returns true if the validationSetSize parameter was updated, false
        otherwise
   */
  bool            setUseValidationSet(const bool useValidationSet);

  /**
     Sets the size of the validation set used by some learning algorithms for
        training. This value represents the percentage of the main
     dataset that will be used for training.  For example, if the
        validationSetSize parameter is 20 then 20% of the training data will be
     used for a validation set leaving 80% of the original data to train the
        model.

     @param validationSetSize: the new validation set size (as a percentage)
     @return returns true if the validationSetSize parameter was updated, false
        otherwise
   */
  bool            setValidationSetSize(const uint32 validationSetSize);

  /**
     Sets if the order of the training dataset should be randomized at each
        epoch of training.
     Randomizing the order of the training dataset stops a learning algorithm
        from focusing too much on the first few examples in the dataset.

     @param randomiseTrainingOrder: if true then the order in which training
        samples are supplied to a learning algorithm will be randomized
     @return returns true if the parameter was updated, false otherwise
   */
  bool            setRandomiseTrainingOrder(const bool randomiseTrainingOrder);


  /**
     Registers the observer with the training result observer manager. The
        observer will then be notified when any new training result is computed.

     @param observer: the observer you want to register with the learning
        algorithm
     @return returns true the observer was added, false otherwise
   */
  bool registerTrainingResultsObserver(
    Observer<TrainingResult>& observer);

  /**
     Registers the observer with the test result observer manager. The observer
        will then be notified when any new test result is computed.

     @param observer: the observer you want to register with the learning
        algorithm
     @return returns true the observer was added, false otherwise
   */
  bool registerTestResultsObserver(
    Observer<TestInstanceResult>& observer);

  /**
     Removes the observer from the training result observer manager.

     @param observer: the observer you want to remove from the learning
        algorithm
     @return returns true if the observer was removed, false otherwise
   */
  bool removeTrainingResultsObserver(
    const Observer<TrainingResult>& observer);

  /**
     Removes the observer from the test result observer manager.

     @param observer: the observer you want to remove from the learning
        algorithm
     @return returns true if the observer was removed, false otherwise
   */
  bool removeTestResultsObserver(
    const Observer<TestInstanceResult>& observer);

  /**
     Removes all observers from the training result observer manager.

     @return returns true if all the observers were removed, false otherwise
   */
  bool                  removeAllTrainingObservers();

  /**
     Removes all observers from the training result observer manager.

     @return returns true if all the observers were removed, false otherwise
   */
  bool                  removeAllTestObservers();

  /**
     Notifies all observers that have subscribed to the training results
        observer manager.

     @param data: stores the training results data for the current update
     @return returns true if all the observers were notified, false otherwise
   */
  bool                  notifyTrainingResultsObservers(const TrainingResult& data);

  /**
     Notifies all observers that have subscribed to the test results observer
        manager.

     @param data: stores the test results data for the current update
     @return returns true if all the observers were notified, false otherwise
   */
  bool                  notifyTestResultsObservers(const TestInstanceResult& data);

  /**
     This functions returns a pointer to the current instance.

     @return returns a MLBase pointer to the current instance.
   */
  MLBase              * getMLBasePointer();

  /**
     This functions returns a const pointer to the current instance.

     @return returns a const MLBase pointer to the current instance.
   */
  const MLBase        * getMLBasePointer() const;

  /**
     Gets the training results from the last training phase. Each element in the
        vector represents the training results from 1 training iteration.

     @return returns a vector of TrainingResult instances containing the
        training results from the most recent training phase
   */
  Vector<TrainingResult>getTrainingResults() const;

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

  bool trained;
  bool useScaling;
  bool converged;
  DataType inputType;
  DataType outputType;
  BaseType baseType;
  uint32   numInputDimensions;
  uint32   numOutputDimensions;
  uint32   numTrainingIterationsToConverge;
  uint32   minNumEpochs;
  uint32   maxNumEpochs;
  uint32   batchSize;
  uint32   validationSetSize;
  uint32   numRestarts;
  float    learningRate;
  float    minChange;
  float    rmsTrainingError;
  float    rmsValidationError;
  float    totalSquaredTrainingError;
  float    validationSetAccuracy;
  bool     useValidationSet;
  bool     randomiseTrainingOrder;
  VectorFloat validationSetPrecision;
  VectorFloat validationSetRecall;
  Random random;
  Vector<TrainingResult> trainingResults;
  TrainingResultsObserverManager trainingResultsObserverManager;
  TestResultsObserverManager     testResultsObserverManager;
};
}
