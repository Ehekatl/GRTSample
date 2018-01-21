#pragma once

#include "../GRT.h"
#include "../Util/MinMax.h"
#include "../Util/Math.h"
#include "Vector.h"
#include <limits>


namespace GRT {
  class GRT_API VectorFloat : public Vector< float > {
  public:
    /**
    Default Constructor
    */
    VectorFloat();

    /**
    Constructor, sets the size of the vector

    @param size: sets the size of the vector
    */
    VectorFloat(const size_type size);

    /**
    Constructor, sets the size of the vector and sets all elements to value

    @param size: sets the size of the vector
    @param value: the value that will be written to all elements in the vector
    */
    VectorFloat(const size_type size, const float &value);

    /**
    Copy Constructor, copies the values from the rhs VectorFloat to this VectorFloat instance

    @param rhs: the VectorFloat from which the values will be copied
    */
    VectorFloat(const VectorFloat &rhs);

    /**
    Destructor, cleans up any memory
    */
    virtual ~VectorFloat();

    /**
    Defines how the data from the rhs VectorFloat should be copied to this VectorFloat

    @param rhs: another instance of a VectorFloat
    @return returns a reference to this instance of the VectorFloat
    */
    VectorFloat& operator=(const VectorFloat &rhs);

    /**
    Defines how the data from the rhs Vector< Float > should be copied to this VectorFloat

    @param rhs: an instance of a Vector< Float >
    @return returns a reference to this instance of the VectorFloat
    */
    VectorFloat& operator=(const Vector< float > &rhs);

    /**
    Scales the vector to a new range given by the min and max targets, this uses the minimum and maximum values in the
    existing vector as the minSource and maxSource for min-max scaling.

    @return returns true if the vector was scaled, false otherwise
    */
    bool scale(const float minTarget, const float maxTarget, const bool constrain = true);

    /**
    Scales the vector to a new range given by the min and max targets using the ranges as the source ranges.

    @return returns true if the vector was scaled, false otherwise
    */
    bool scale(const float minSource, const float maxSource, const float minTarget, const float maxTarget, const bool constrain = true);

    /**
    @return returns the minimum value in the vector
    */
    float getMinValue() const;

    /**
    @return returns the maximum value in the vector
    */
    float getMaxValue() const;

    /**
    @return returns the mean of the vector
    */
    float getMean() const;

    /**
    @return returns the standard deviation of the vector
    */
    float getStdDev() const;

    /**
    @return returns the minimum and maximum values in the vector
    */
    MinMax getMinMax() const;
  };
}