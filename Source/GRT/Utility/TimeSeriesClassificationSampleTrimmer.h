#pragma once

#include "../GRT.h"
#include "../Types/TimeSeriesClassificationData.h"

namespace GRT {
class GRT_API TimeSeriesClassificationSampleTrimmer {
public:

  /**
     Default Constructor.
   */
  TimeSeriesClassificationSampleTrimmer(float trimThreshold = 0.1,
                                        float maximumTrimPercentage = 80);

  /**
     Default Destructor
   */
  ~TimeSeriesClassificationSampleTrimmer();

  /**
     Defines the equals operator. Copies the settings from the rhs instance to
        this instance

     @param rhs: the instance from which the settings will be copied
     @return returns a reference to this instance
   */
  TimeSeriesClassificationSampleTrimmer& operator=(
    const TimeSeriesClassificationSampleTrimmer& rhs) {
    if (this != &rhs) {
      this->trimThreshold         = rhs.trimThreshold;
      this->maximumTrimPercentage = rhs.maximumTrimPercentage;
    }
    return *this;
  }

  /**
     The function attempts to detect and remove these static areas of data. This
        is done by computing the summed absolute energy of the
     timeseries data, normalizing the energy profile by the maximum energy
        value, and then searching for areas at the start and end of
     the timeseries that are below a specific trimthreshold (set by the user).

     Any data that is below the trimthreshold will be removed, up until the
        first value that exceeds the threshold.  This search is
     run both from the start of the timeseries (searching forward) and the end
        of the timeseries (searching backwards).  If the length
     of the new timeseries is below the maximumTrimPercentage, then the
        timeseries will be trimmed and the trimTimeSeries function
     will return true.  If the length of the new is above the
        maximumTrimPercentage, then the timeseries will not be trimmed and the
     trimTimeSeries function will return false. Set the maximumTrimPercentage to
        100 if you want the timeseries to always be trimmed.

     @param timeSeries: the timeseries to be trimmed (will be trimmed in place)
     @return returns true if the timeseries was trimmed, false otherwise
   */
  bool trimTimeSeries(TimeSeriesClassificationSample& timeSeries);

protected:

  float trimThreshold;
  float maximumTrimPercentage;
};
}
