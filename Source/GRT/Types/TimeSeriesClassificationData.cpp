#include "../GRT.h"
#include "TimeSeriesClassificationData.h"

namespace GRT {
TimeSeriesClassificationData::TimeSeriesClassificationData(
  uint32  numDimensions,
  FString datasetName,
  FString infoText) {
  this->numDimensions   = numDimensions;
  this->datasetName     = datasetName;
  this->infoText        = infoText;
  totalNumSamples       = 0;
  crossValidationSetup  = false;
  useExternalRanges     = false;
  allowNullGestureClass = true;

  if (numDimensions > 0) {
    setNumDimensions(numDimensions);
  }
}

TimeSeriesClassificationData::TimeSeriesClassificationData(
  const TimeSeriesClassificationData& rhs) {
  *this = rhs;
}

TimeSeriesClassificationData::~TimeSeriesClassificationData() {}

TimeSeriesClassificationData& TimeSeriesClassificationData::operator=(
  const TimeSeriesClassificationData& rhs) {
  if (this != &rhs) {
    this->datasetName           = rhs.datasetName;
    this->infoText              = rhs.infoText;
    this->numDimensions         = rhs.numDimensions;
    this->useExternalRanges     = rhs.useExternalRanges;
    this->allowNullGestureClass = rhs.allowNullGestureClass;
    this->crossValidationSetup  = rhs.crossValidationSetup;
    this->crossValidationIndexs = rhs.crossValidationIndexs;
    this->totalNumSamples       = rhs.totalNumSamples;
    this->data                  = rhs.data;
    this->classTracker          = rhs.classTracker;
    this->externalRanges        = rhs.externalRanges;
  }
  return *this;
}

void TimeSeriesClassificationData::clear() {
  totalNumSamples = 0;
  data.clear();
  classTracker.clear();
}

bool TimeSeriesClassificationData::setNumDimensions(const uint32 l_numDimensions)
{
  if (l_numDimensions > 0) {
    // Clear any previous training data
    clear();

    // Set the dimensionality of the training data
    this->numDimensions = l_numDimensions;

    useExternalRanges = false;
    externalRanges.clear();

    return true;
  }

  UE_LOG(GRTModule, Error,
         TEXT(
           "setNumDimensions(uint32 numDimensions) - The number of dimensions of the dataset must be greater than zero!"));
  return false;
}

bool TimeSeriesClassificationData::setDatasetName(const FString l_datasetName) {
  // Make sure there are no spaces in the FString
  if (!l_datasetName.Contains(" ")) {
    this->datasetName = l_datasetName;
    return true;
  }

  UE_LOG(GRTModule, Error,
         TEXT(
           "setDatasetName(FString datasetName) - The dataset name cannot contain any spaces!"));
  return false;
}

bool TimeSeriesClassificationData::setInfoText(const FString l_infoText) {
  this->infoText = l_infoText;
  return true;
}

bool TimeSeriesClassificationData::setClassNameForCorrespondingClassLabel(
  const FString className,
  const uint32  classLabel) {
  for (uint32 i = 0; i < classTracker.size(); i++) {
    if (classTracker[i].classLabel == classLabel) {
      classTracker[i].className = className;
      return true;
    }
  }

  return false;
}

bool TimeSeriesClassificationData::setAllowNullGestureClass(
  const bool l_allowNullGestureClass) {
  this->allowNullGestureClass = l_allowNullGestureClass;
  return true;
}

bool TimeSeriesClassificationData::addSample(const uint32       classLabel,
                                             const MatrixFloat& trainingSample) {
  if (trainingSample.getNumCols() != numDimensions) {
    UE_LOG(GRTModule, Error,
           TEXT(
             "addSample(uint32 classLabel, MatrixFloat trainingSample) - The dimensionality of the training sample ( %d ) does not match that of the dataset ( %d )"),
           (uint32)trainingSample.getNumCols(), numDimensions);
    return false;
  }

  // The class label must be greater than zero (as zero is used for the null
  // rejection class label
  if ((classLabel == GRT_DEFAULT_NULL_CLASS_LABEL) && !allowNullGestureClass) {
    UE_LOG(GRTModule, Error,
           TEXT(
             "addSample(uint32 classLabel, MatrixFloat sample) - the class label can not be 0!"));
    return false;
  }

  TimeSeriesClassificationSample newSample(classLabel, trainingSample);
  data.push_back(newSample);
  totalNumSamples++;

  if (classTracker.size() == 0) {
    ClassTracker tracker(classLabel, 1);
    classTracker.push_back(tracker);
  }
  else {
    bool labelFound = false;

    for (uint32 i = 0; i < classTracker.size(); i++) {
      if (classLabel == classTracker[i].classLabel) {
        classTracker[i].counter++;
        labelFound = true;
        break;
      }
    }

    if (!labelFound) {
      ClassTracker tracker(classLabel, 1);
      classTracker.push_back(tracker);
    }
  }
  return true;
}

uint32 TimeSeriesClassificationData::eraseAllSamplesWithClassLabel(
  const uint32 classLabel) {
  uint32 numExamplesRemoved  = 0;
  uint32 numExamplesToRemove = 0;

  // Find out how many training examples we need to remove
  for (uint32 i = 0; i < classTracker.size(); i++) {
    if (classTracker[i].classLabel == classLabel) {
      numExamplesToRemove = classTracker[i].counter;
      classTracker.erase(classTracker.begin() + i);
      break;
    }
  }

  // Remove the samples with the matching class ID
  if (numExamplesToRemove > 0) {
    uint32 i = 0;

    while (numExamplesRemoved < numExamplesToRemove) {
      if (data[i].getClassLabel() == classLabel) {
        data.erase(data.begin() + i);
        numExamplesRemoved++;
      }
      else if (++i == data.size()) break;
    }
  }

  totalNumSamples = (uint32)data.size();

  return numExamplesRemoved;
}

bool TimeSeriesClassificationData::removeLastSample() {
  if (totalNumSamples > 0) {
    // Find the corresponding class ID for the last training example
    uint32 classLabel = data[totalNumSamples - 1].getClassLabel();

    // Remove the training example from the buffer
    data.erase(data.end() - 1);

    totalNumSamples = (uint32)data.size();

    // Remove the value from the counter
    for (uint32 i = 0; i < classTracker.size(); i++) {
      if (classTracker[i].classLabel == classLabel) {
        classTracker[i].counter--;
        break;
      }
    }

    return true;
  }
  else return false;
}

bool TimeSeriesClassificationData::relabelAllSamplesWithClassLabel(
  const uint32 oldClassLabel,
  const uint32 newClassLabel) {
  bool   oldClassLabelFound          = false;
  bool   newClassLabelAllReadyExists = false;
  uint32 indexOfOldClassLabel        = 0;
  uint32 indexOfNewClassLabel        = 0;

  // Find out how many training examples we need to relabel
  for (uint32 i = 0; i < classTracker.size(); i++) {
    if (classTracker[i].classLabel == oldClassLabel) {
      indexOfOldClassLabel = i;
      oldClassLabelFound   = true;
    }

    if (classTracker[i].classLabel == newClassLabel) {
      indexOfNewClassLabel        = i;
      newClassLabelAllReadyExists = true;
    }
  }

  // If the old class label was not found then we can't do anything
  if (!oldClassLabelFound) {
    return false;
  }

  // Relabel the old class labels
  for (uint32 i = 0; i < totalNumSamples; i++) {
    if (data[i].getClassLabel() == oldClassLabel) {
      data[i].setTrainingSample(newClassLabel, data[i].getData());
    }
  }

  // Update the class label counters
  if (newClassLabelAllReadyExists) {
    // Add the old sample count to the new sample count
    classTracker[indexOfNewClassLabel].counter +=
      classTracker[indexOfOldClassLabel].counter;

    // Erase the old class tracker
    classTracker.erase(classTracker.begin() + indexOfOldClassLabel);
  }
  else {
    // Create a new class tracker
    classTracker.push_back(ClassTracker(newClassLabel,
                                        classTracker[indexOfOldClassLabel].counter,
                                        classTracker[indexOfOldClassLabel].
                                        className));
  }

  return true;
}

bool TimeSeriesClassificationData::setExternalRanges(
  const Vector<MinMax>& l_externalRanges,
  const bool            l_useExternalRanges) {
  if (l_externalRanges.size() != numDimensions) return false;

  this->externalRanges    = l_externalRanges;
  this->useExternalRanges = l_useExternalRanges;

  return true;
}

bool TimeSeriesClassificationData::enableExternalRangeScaling(
  const bool l_useExternalRanges) {
  if (externalRanges.size() == numDimensions) {
    this->useExternalRanges = l_useExternalRanges;
    return true;
  }
  return false;
}

bool TimeSeriesClassificationData::scale(const float minTarget,
                                         const float maxTarget) {
  Vector<MinMax> ranges = getRanges();
  return scale(ranges, minTarget, maxTarget);
}

bool TimeSeriesClassificationData::scale(const Vector<MinMax>& ranges,
                                         const float           minTarget,
                                         const float           maxTarget) {
  if (ranges.size() != numDimensions) return false;

  // Scale the training data
  for (uint32 i = 0; i < totalNumSamples; i++) {
    for (uint32 x = 0; x < data[i].getLength(); x++) {
      for (uint32 j = 0; j < numDimensions; j++) {
        data[i][x][j] = grt_scale(data[i][x][j],
                                  ranges[j].minValue,
                                  ranges[j].maxValue,
                                  minTarget,
                                  maxTarget);
      }
    }
  }

  return true;
}

bool TimeSeriesClassificationData::save(const FString& filename) const {
  // Save it as a custom GRT file
  return saveDatasetToFile(filename);
}

bool TimeSeriesClassificationData::load(const FString& filename) {
  // load it as a custom GRT file
  return loadDatasetFromFile(filename);
}

bool TimeSeriesClassificationData::saveDatasetToFile(const FString fileName)
const {
  std::fstream file;

  file.open(TCHAR_TO_UTF8(*fileName), std::ios::out);

  if (!file.is_open()) {
    UE_LOG(GRTModule, Error,
           TEXT("saveDatasetToFile(FString fileName) -  Failed to open file!"));
    return false;
  }

  file << "GRT_LABELLED_TIME_SERIES_CLASSIFICATION_DATA_FILE_V1.0\n";
  file << "DatasetName: " << TCHAR_TO_UTF8(*datasetName) << std::endl;
  file << "InfoText: " << TCHAR_TO_UTF8(*infoText) << std::endl;
  file << "NumDimensions: " << numDimensions << std::endl;
  file << "TotalNumTrainingExamples: " << totalNumSamples << std::endl;
  file << "NumberOfClasses: " << classTracker.size() << std::endl;
  file << "ClassIDsAndCounters: " << std::endl;

  for (uint32 i = 0; i < classTracker.size(); i++) {
    file << classTracker[i].classLabel << "\t" << classTracker[i].counter <<
      std::endl;
  }

  file << "UseExternalRanges: " << useExternalRanges << std::endl;

  if (useExternalRanges) {
    for (uint32 i = 0; i < externalRanges.size(); i++) {
      file << externalRanges[i].minValue << "\t" << externalRanges[i].maxValue <<
        std::endl;
    }
  }

  file << "LabelledTimeSeriesTrainingData:\n";

  for (uint32 x = 0; x < totalNumSamples; x++) {
    file << "************TIME_SERIES************\n";
    file << "ClassID: " << data[x].getClassLabel() << std::endl;
    file << "TimeSeriesLength: " << data[x].getLength() << std::endl;
    file << "TimeSeriesData: \n";

    for (uint32 i = 0; i < data[x].getLength(); i++) {
      for (uint32 j = 0; j < numDimensions; j++) {
        file << data[x][i][j];

        if (j < numDimensions - 1) file << "\t";
      } file << std::endl;
    }
  }

  file.close();
  return true;
}

bool TimeSeriesClassificationData::loadDatasetFromFile(const FString filename)
{
  std::fstream file;

  file.open(TCHAR_TO_UTF8(*filename), std::ios::in);
  uint32 numClasses = 0;
  clear();

  if (!file.is_open()) {
    UE_LOG(GRTModule, Error,
           TEXT("loadDatasetFromFile(FString filename) - FILE NOT OPEN!"));
    return false;
  }

  std::string word;
  std::string std_datasetName;


  // Check to make sure this is a file with the Training File Format
  file >> word;

  if (word != "GRT_LABELLED_TIME_SERIES_CLASSIFICATION_DATA_FILE_V1.0") {
    file.close();
    clear();
    UE_LOG(GRTModule, Error,
           TEXT(
             "loadDatasetFromFile(FString filename) - Failed to find file header!"));
    return false;
  }

  // Get the name of the dataset
  file >> word;

  if (word != "DatasetName:") {
    UE_LOG(GRTModule, Error,
           TEXT(
             "loadDatasetFromFile(FString filename) - failed to find DatasetName!"));
    file.close();
    return false;
  }
  file >> std_datasetName;
  datasetName = FString(std_datasetName.c_str());

  file >> word;

  if (word != "InfoText:") {
    UE_LOG(GRTModule, Error,
           TEXT(
             "loadDatasetFromFile(FString filename) - failed to find InfoText!"));
    file.close();
    return false;
  }

  // Load the info text
  file >> word;
  infoText = "";

  while (word != "NumDimensions:") {
    infoText += FString(word.c_str()) + " ";
    file >> word;
  }

  // Get the number of dimensions in the training data
  if (word != "NumDimensions:") {
    file.close();
    clear();
    UE_LOG(GRTModule, Error,
           TEXT(
             "loadDatasetFromFile(FString filename) - Failed to find NumDimensions!"));
    return false;
  }
  file >> numDimensions;

  // Get the total number of training examples in the training data
  file >> word;

  if (word != "TotalNumTrainingExamples:") {
    file.close();
    clear();
    UE_LOG(GRTModule, Error,
           TEXT(
             "loadDatasetFromFile(FString filename) - Failed to find TotalNumTrainingExamples!"));
    return false;
  }
  file >> totalNumSamples;

  // Get the total number of classes in the training data
  file >> word;

  if (word != "NumberOfClasses:") {
    file.close();
    clear();
    UE_LOG(GRTModule, Error,
           TEXT(
             "loadDatasetFromFile(FString filename) - Failed to find NumberOfClasses!"));
    return false;
  }
  file >> numClasses;

  // Resize the class counter buffer and load the counters
  classTracker.resize(numClasses);

  // Get the total number of classes in the training data
  file >> word;

  if (word != "ClassIDsAndCounters:") {
    file.close();
    clear();
    UE_LOG(GRTModule, Error,
           TEXT(
             "loadDatasetFromFile(FString filename) - Failed to find ClassIDsAndCounters!"));
    return false;
  }

  for (uint32 i = 0; i < classTracker.size(); i++) {
    file >> classTracker[i].classLabel;
    file >> classTracker[i].counter;
  }

  // Get the UseExternalRanges
  file >> word;

  if (word != "UseExternalRanges:") {
    file.close();
    clear();
    UE_LOG(GRTModule, Error,
           TEXT(
             "loadDatasetFromFile(FString filename) - Failed to find UseExternalRanges!"));
    return false;
  }

  file >> useExternalRanges;

  if (useExternalRanges) {
    externalRanges.resize(numDimensions);

    for (uint32 i = 0; i < externalRanges.size(); i++) {
      file >> externalRanges[i].minValue;
      file >> externalRanges[i].maxValue;
    }
  }

  // Get the main training data
  file >> word;

  if (word != "LabelledTimeSeriesTrainingData:") {
    file.close();
    clear();
    UE_LOG(GRTModule, Error,
           TEXT(
             "loadDatasetFromFile(FString filename) - Failed to find LabelledTimeSeriesTrainingData!"));
    return false;
  }

  // Reset the memory
  data.resize(totalNumSamples, TimeSeriesClassificationSample());

  // Load each of the time series
  for (uint32 x = 0; x < totalNumSamples; x++) {
    uint32 classLabel       = 0;
    uint32 timeSeriesLength = 0;

    file >> word;

    if (word != "************TIME_SERIES************") {
      file.close();
      clear();
      UE_LOG(GRTModule, Error,
             TEXT(
               "loadDatasetFromFile(FString filename) - Failed to find TimeSeries Header!"));
      return false;
    }

    file >> word;

    if (word != "ClassID:") {
      file.close();
      clear();
      UE_LOG(GRTModule, Error,
             TEXT(
               "loadDatasetFromFile(FString filename) - Failed to find ClassID!"));
      return false;
    }
    file >> classLabel;

    file >> word;

    if (word != "TimeSeriesLength:") {
      file.close();
      clear();
      UE_LOG(GRTModule, Error,
             TEXT(
               "loadDatasetFromFile(FString filename) - Failed to find TimeSeriesLength!"));
      return false;
    }
    file >> timeSeriesLength;

    file >> word;

    if (word != "TimeSeriesData:") {
      file.close();
      clear();
      UE_LOG(GRTModule, Error,
             TEXT(
               "loadDatasetFromFile(FString filename) - Failed to find TimeSeriesData!"));
      return false;
    }

    // Load the time series data
    MatrixFloat trainingExample(timeSeriesLength, numDimensions);

    for (uint32 i = 0; i < timeSeriesLength; i++) {
      for (uint32 j = 0; j < numDimensions; j++) {
        file >> trainingExample[i][j];
      }
    }

    data[x].setTrainingSample(classLabel, trainingExample);
  }

  file.close();
  return true;
}

FString TimeSeriesClassificationData::getStatsAsString() const {
  FString stats;

  stats += "DatasetName:\t" + datasetName + "\n";
  stats += "DatasetInfo:\t" + infoText + "\n";
  stats += "Number of Dimensions:\t" + Util::toString(numDimensions) + "\n";
  stats += "Number of Samples:\t" + Util::toString(totalNumSamples) + "\n";
  stats += "Number of Classes:\t" + Util::toString(getNumClasses()) + "\n";
  stats += "ClassStats:\n";

  for (uint32 k = 0; k < getNumClasses(); k++) {
    stats += "ClassLabel:\t" + Util::toString(classTracker[k].classLabel);
    stats += "\tNumber of Samples:\t" + Util::toString(classTracker[k].counter);
    stats += "\tClassName:\t" + classTracker[k].className + "\n";
  }

  Vector<MinMax> ranges = getRanges();

  stats += "Dataset Ranges:\n";

  for (uint32 j = 0; j < ranges.size(); j++) {
    stats += "[" + Util::toString(j + 1) + "] Min:\t" + Util::toString(
      ranges[j].minValue) + "\tMax: " + Util::toString(ranges[j].maxValue) + "\n";
  }

  stats += "Timeseries Lengths:\n";
  uint32 M = (uint32)data.size();

  for (uint32 j = 0; j < M; j++) {
    stats += "ClassLabel: " + Util::toString(data[j].getClassLabel()) +
             " Length:\t" + Util::toString(data[j].getLength()) + "\n";
  }

  return stats;
}

TimeSeriesClassificationData TimeSeriesClassificationData::split(
  const uint32 trainingSizePercentage,
  const bool   useStratifiedSampling) {
  // Partitions the dataset into a training dataset (which is kept by this
  // instance of the TimeSeriesClassificationData) and
  // a testing/validation dataset (which is return as a new instance of the
  // TimeSeriesClassificationData).  The trainingSizePercentage
  // therefore sets the size of the data which remains in this instance and the
  // remaining percentage of data is then added to
  // the testing/validation dataset

  // The dataset has changed so flag that any previous cross validation setup
  // will now not work
  crossValidationSetup = false;
  crossValidationIndexs.clear();

  TimeSeriesClassificationData trainingSet(numDimensions);
  TimeSeriesClassificationData testSet(numDimensions);
  trainingSet.setAllowNullGestureClass(allowNullGestureClass);
  testSet.setAllowNullGestureClass(allowNullGestureClass);
  Vector<uint32> indexs(totalNumSamples);

  // Create the random partion indexs
  Random random;
  uint32 randomIndex = 0;

  if (useStratifiedSampling) {
    // Break the data into seperate classes
    Vector<Vector<uint32> > classData(getNumClasses());

    // Add the indexs to their respective classes
    for (uint32 i = 0; i < totalNumSamples; i++) {
      classData[getClassLabelIndexValue(data[i].getClassLabel())].push_back(i);
    }

    // Randomize the order of the indexs in each of the class index buffers
    for (uint32 k = 0; k < getNumClasses(); k++) {
      uint32 numSamples = (uint32)classData[k].size();

      for (uint32 x = 0; x < numSamples; x++) {
        // Pick a random index
        randomIndex = random.getRandomNumberInt(0, numSamples);

        // Swap the indexs
        SWAP(classData[k][x], classData[k][randomIndex]);
      }
    }

    // Loop over each class and add the data to the trainingSet and testSet
    for (uint32 k = 0; k < getNumClasses(); k++) {
      uint32 numTrainingExamples = (uint32)floor(float(
                                                   classData[k].size()) / 100.0 *
                                                 float(trainingSizePercentage));

      // Add the data to the training and test sets
      for (uint32 i = 0; i < numTrainingExamples; i++) {
        trainingSet.addSample(data[classData[k][i]].getClassLabel(),
                              data[classData[k][i]].getData());
      }

      for (uint32 i = numTrainingExamples; i < classData[k].size(); i++) {
        testSet.addSample(data[classData[k][i]].getClassLabel(),
                          data[classData[k][i]].getData());
      }
    }

    // Overwrite the training data in this instance with the training data of
    // the trainingSet
    data            = trainingSet.getClassificationData();
    totalNumSamples = trainingSet.getNumSamples();
  }
  else {
    const uint32 numTrainingExamples = (uint32)floor(float(
                                                       totalNumSamples) / 100.0 *
                                                     float(trainingSizePercentage));

    for (uint32 i = 0; i < totalNumSamples; i++) indexs[i] = i;

    for (uint32 x = 0; x < totalNumSamples; x++) {
      // Pick a random index
      randomIndex = random.getRandomNumberInt(0, totalNumSamples);

      // Swap the indexs
      SWAP(indexs[x], indexs[randomIndex]);
    }

    // Add the data to the training and test sets
    for (uint32 i = 0; i < numTrainingExamples; i++) {
      trainingSet.addSample(
        data[indexs[i]].getClassLabel(), data[indexs[i]].getData());
    }

    for (uint32 i = numTrainingExamples; i < totalNumSamples; i++) {
      testSet.addSample(data[indexs[i]].getClassLabel(),
                        data[indexs[i]].getData());
    }

    // Overwrite the training data in this instance with the training data of
    // the trainingSet
    data            = trainingSet.getClassificationData();
    totalNumSamples = trainingSet.getNumSamples();
  }

  return testSet;
}

bool TimeSeriesClassificationData::merge(
  const TimeSeriesClassificationData& labelledData) {
  if (labelledData.getNumDimensions() != numDimensions) {
    UE_LOG(GRTModule, Error,
           TEXT(
             "merge(TimeSeriesClassificationData &labelledData) - The number of dimensions in the labelledData (%d) does not match the number of dimensions of this dataset (%d)"),
           labelledData.getNumDimensions(), numDimensions);
    return false;
  }

  // The dataset has changed so flag that any previous cross validation setup
  // will now not work
  crossValidationSetup = false;
  crossValidationIndexs.clear();

  // Add the data from the labelledData to this instance
  for (uint32 i = 0; i < labelledData.getNumSamples(); i++) {
    addSample(labelledData[i].getClassLabel(), labelledData[i].getData());
  }

  // Set the class names from the dataset
  Vector<ClassTracker> l_classTracker = labelledData.getClassTracker();

  for (uint32 i = 0; i < l_classTracker.size(); i++) {
    setClassNameForCorrespondingClassLabel(l_classTracker[i].className,
                                           l_classTracker[i].classLabel);
  }

  return true;
}

bool TimeSeriesClassificationData::spiltDataIntoKFolds(const uint32 K,
                                                       const bool   useStratifiedSampling)
{
  crossValidationSetup = false;
  crossValidationIndexs.clear();

  // K can not be zero
  if (K == 0) {
    UE_LOG(GRTModule, Error,
           TEXT("spiltDataIntoKFolds(uint32 K) - K can not be zero!"));
    return false;
  }

  // K can not be larger than the number of examples
  if (K > totalNumSamples) {
    UE_LOG(GRTModule, Error,
           TEXT(
             "spiltDataIntoKFolds(uint32 K,bool useStratifiedSampling) - K can not be larger than the total number of samples in the dataset!"));
    return false;
  }

  // K can not be larger than the number of examples in a specific class if the
  // stratified sampling option is true
  if (useStratifiedSampling) {
    for (uint32 c = 0; c < classTracker.size(); c++) {
      if (K > classTracker[c].counter) {
        UE_LOG(GRTModule, Error,
               TEXT(
                 "spiltDataIntoKFolds(uint32 K,bool useStratifiedSampling) - K can not be larger than the number of samples in any given class!"));
        return false;
      }
    }
  }

  // Setup the dataset for k-fold cross validation
  kFoldValue = K;
  Vector<uint32> indexs(totalNumSamples);

  // Work out how many samples are in each fold, the last fold might have more
  // samples than the others
  uint32 numSamplesPerFold = (uint32)floor(totalNumSamples / float(K));

  // Resize the cross validation indexs buffer
  crossValidationIndexs.resize(K);

  // Create the random partion indexs
  Random random;
  uint32 randomIndex = 0;

  if (useStratifiedSampling) {
    // Break the data into seperate classes
    Vector<Vector<uint32> > classData(getNumClasses());

    // Add the indexs to their respective classes
    for (uint32 i = 0; i < totalNumSamples; i++) {
      classData[getClassLabelIndexValue(data[i].getClassLabel())].push_back(i);
    }

    // Randomize the order of the indexs in each of the class index buffers
    for (uint32 c = 0; c < getNumClasses(); c++) {
      uint32 numSamples = (uint32)classData[c].size();

      for (uint32 x = 0; x < numSamples; x++) {
        // Pick a random index
        randomIndex = random.getRandomNumberInt(0, numSamples);

        // Swap the indexs
        SWAP(classData[c][x], classData[c][randomIndex]);
      }
    }

    // Loop over each of the classes and add the data equally to each of the k
    // folds until there is no data left
    Vector<uint32>::iterator iter;

    for (uint32 c = 0; c < getNumClasses(); c++) {
      iter = classData[c].begin();
      uint32 k = 0;

      while (iter != classData[c].end()) {
        crossValidationIndexs[k].push_back(*iter);
        iter++;
        k++;
        k = k % K;
      }
    }
  }
  else {
    // Randomize the order of the data
    for (uint32 i = 0; i < totalNumSamples; i++) indexs[i] = i;

    for (uint32 x = 0; x < totalNumSamples; x++) {
      // Pick a random index
      randomIndex = random.getRandomNumberInt(0, totalNumSamples);

      // Swap the indexs
      SWAP(indexs[x], indexs[randomIndex]);
    }

    uint32 counter   = 0;
    uint32 foldIndex = 0;

    for (uint32 i = 0; i < totalNumSamples; i++) {
      // Add the index to the current fold
      crossValidationIndexs[foldIndex].push_back(indexs[i]);

      // Move to the next fold if ready
      if ((++counter == numSamplesPerFold) && (foldIndex < K - 1)) {
        foldIndex++;
        counter = 0;
      }
    }
  }

  crossValidationSetup = true;
  return true;
}

TimeSeriesClassificationData TimeSeriesClassificationData::getTrainingFoldData(
  const uint32 foldIndex) const {
  TimeSeriesClassificationData trainingData;

  if (!crossValidationSetup) {
    UE_LOG(GRTModule, Error,
           TEXT(
             "getTrainingFoldData(uint32 foldIndex) - Cross Validation has not been setup! You need to call the spiltDataIntoKFolds(uint32 K,bool useStratifiedSampling) function first before calling this function!"));
    return trainingData;
  }

  if (foldIndex >= kFoldValue) return trainingData;

  trainingData.setNumDimensions(numDimensions);

  // Add the data to the training set, this will consist of all the data that is
  // NOT in the foldIndex
  uint32 index = 0;

  for (uint32 k = 0; k < kFoldValue; k++) {
    if (k != foldIndex) {
      for (uint32 i = 0; i < crossValidationIndexs[k].size(); i++) {
        index = crossValidationIndexs[k][i];
        trainingData.addSample(data[index].getClassLabel(),
                               data[index].getData());
      }
    }
  }

  return trainingData;
}

TimeSeriesClassificationData TimeSeriesClassificationData::getTestFoldData(
  const uint32 foldIndex) const {
  TimeSeriesClassificationData testData;

  if (!crossValidationSetup) return testData;

  if (foldIndex >= kFoldValue) return testData;

  // Add the data to the training
  testData.setNumDimensions(numDimensions);

  uint32 index = 0;

  for (uint32 i = 0; i < crossValidationIndexs[foldIndex].size(); i++) {
    index = crossValidationIndexs[foldIndex][i];
    testData.addSample(data[index].getClassLabel(), data[index].getData());
  }

  return testData;
}

TimeSeriesClassificationData TimeSeriesClassificationData::getClassData(
  const uint32 classLabel) const {
  TimeSeriesClassificationData classData(numDimensions);

  for (uint32 x = 0; x < totalNumSamples; x++) {
    if (data[x].getClassLabel() == classLabel) {
      classData.addSample(classLabel, data[x].getData());
    }
  }
  return classData;
}

uint32 TimeSeriesClassificationData::getMinimumClassLabel() const {
  uint32 minClassLabel = 99999;

  for (uint32 i = 0; i < classTracker.size(); i++) {
    if (classTracker[i].classLabel < minClassLabel) {
      minClassLabel = classTracker[i].classLabel;
    }
  }

  return minClassLabel;
}

uint32 TimeSeriesClassificationData::getMaximumClassLabel() const {
  uint32 maxClassLabel = 0;

  for (uint32 i = 0; i < classTracker.size(); i++) {
    if (classTracker[i].classLabel > maxClassLabel) {
      maxClassLabel = classTracker[i].classLabel;
    }
  }

  return maxClassLabel;
}

uint32 TimeSeriesClassificationData::getClassLabelIndexValue(
  const uint32 classLabel)
const {
  for (uint32 k = 0; k < classTracker.size(); k++) {
    if (classTracker[k].classLabel == classLabel) {
      return k;
    }
  }
  UE_LOG(GRTModule, Warning,
         TEXT(
           "getClassLabelIndexValue(uint32 classLabel) - Failed to find class label: %d in class tracker!"),
         classLabel);
  return 0;
}

FString TimeSeriesClassificationData::getClassNameForCorrespondingClassLabel(
  const uint32 classLabel) const {
  for (uint32 i = 0; i < classTracker.size(); i++) {
    if (classTracker[i].classLabel == classLabel) {
      return classTracker[i].className;
    }
  }
  return "CLASS_LABEL_NOT_FOUND";
}

Vector<MinMax>TimeSeriesClassificationData::getRanges() const {
  if (useExternalRanges) return externalRanges;

  Vector<MinMax> ranges(numDimensions);

  if (totalNumSamples > 0) {
    for (uint32 j = 0; j < numDimensions; j++) {
      ranges[j].minValue = data[0][0][0];
      ranges[j].maxValue = data[0][0][0];

      for (uint32 x = 0; x < totalNumSamples; x++) {
        for (uint32 i = 0; i < data[x].getLength(); i++) {
          if (data[x][i][j] <
              ranges[j].minValue)  ranges[j].minValue = data[x][i][j];      // Search
                                                                            // for
                                                                            // the
                                                                            // min
                                                                            // value
          else if (data[x][i][j] >
                   ranges[j].maxValue)  ranges[j].maxValue = data[x][i][j]; // Search
                                                                            // for
                                                                            // the
                                                                            // max
                                                                            // value
        }
      }
    }
  }
  return ranges;
}

MatrixFloat TimeSeriesClassificationData::getDataAsMatrixFloat() const {
  // Count how many samples are in the entire dataset
  uint32 M     = 0;
  uint32 index = 0;

  for (uint32 x = 0; x < totalNumSamples; x++) {
    M += data[x].getLength();
  }

  if (M == 0) MatrixFloat();

  // Get all the data and concatenate it into 1 matrix
  MatrixFloat matrixData(M, numDimensions);

  for (uint32 x = 0; x < totalNumSamples; x++) {
    for (uint32 i = 0; i < data[x].getLength(); i++) {
      for (uint32 j = 0; j < numDimensions; j++) {
        matrixData[index][j] = data[x][i][j];
      }
      index++;
    }
  }
  return matrixData;
}
}
