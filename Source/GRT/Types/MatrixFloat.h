#pragma once

#include "../GRT.h"
#include  "Matrix.h"
#include "VectorFloat.h"

namespace GRT {
class GRT_API MatrixFloat : public Matrix<float>{
public:

  /**
     Default Constructor
   */
  MatrixFloat();

  /**
     Constructor, sets the size of the matrix to [rows cols]

     @param rows: sets the number of rows in the matrix, must be a value greater
        than zero
     @param cols: sets the number of columns in the matrix, must be a value
        greater than zero
   */
  MatrixFloat(const uint32 rows,
              const uint32 cols);

  /**
     Copy Constructor, copies the values from the rhs MatrixFloat to this
        MatrixFloat instance

     @param rhs: the MatrixFloat from which the values will be copied
   */
  MatrixFloat(const MatrixFloat& rhs);

  /**
     Copy Constructor, copies the values from the rhs Matrix to this MatrixFloat
        instance

     @param rhs: the Matrix from which the values will be copied
   */
  MatrixFloat(const Matrix<float>& rhs);

  /**
     Copy Constructor, copies the values from the rhs vector to this MatrixFloat
        instance

     @param rhs: a vector of vectors the values will be copied
   */
  MatrixFloat(const Vector<VectorFloat>& rhs);

  /**
     Destructor, cleans up any memory
   */
  virtual ~MatrixFloat();

  /**
     Defines how the data from the rhs MatrixFloat should be copied to this
        MatrixFloat

     @param rhs: another instance of a MatrixFloat
     @return returns a reference to this instance of the MatrixFloat
   */
  MatrixFloat& operator=(const MatrixFloat& rhs);

  /**
     Defines how the data from the rhs Matrix< Float > should be copied to this
        MatrixFloat

     @param rhs: an instance of a Matrix< Float >
     @return returns a reference to this instance of the MatrixFloat
   */
  MatrixFloat& operator=(const Matrix<float>& rhs);

  /**
     Defines how the data from the rhs Vector of VectorFloats should be copied
        to this MatrixFloat

     @param rhs: a Vector of VectorFloats
     @return returns a reference to this instance of the MatrixFloat
   */
  MatrixFloat& operator=(const Vector<VectorFloat>& rhs);

  /**
     Gets a row vector [1 cols] from the Matrix at the row index r

     @param r: the index of the row, this should be in the range [0 rows-1]
     @return returns a row vector from the Matrix at the row index r
   */
  VectorFloat  getRow(const uint32 r) const {
    VectorFloat rowVector(cols);

    for (uint32 c = 0; c < cols; c++) rowVector[c] = dataPtr[r * cols + c];
    return rowVector;
  }

  /**
     Gets a column vector [rows 1] from the Matrix at the column index c

     @param c: the index of the column, this should be in the range [0 cols-1]
     @return returns a column vector from the Matrix at the column index c
   */
  VectorFloat getCol(const uint32 c) const {
    VectorFloat columnVector(rows);

    for (uint32 r = 0; r < rows; r++) columnVector[r] = dataPtr[r * cols + c];
    return columnVector;
  }

  /**
     Transposes the data.

     @return returns true or false, indicating if the transpose was successful
   */
  bool transpose();

  /**
     Scales the matrix to a new range given by the min and max targets.

     @return returns true if the matrix was scaled, false otherwise
   */
  bool scale(const float minTarget,
             const float maxTarget);

  /**
     Scales the matrix to a new range given by the min and max targets using the
        ranges as the source ranges.

     @return returns true if the matrix was scaled, false otherwise
   */
  bool scale(const Vector<MinMax>& ranges,
             const float           minTarget,
             const float           maxTarget);

  /**
     Normalizes each row in the matrix by subtracting the row mean and dividing
        by the row standard deviation.
     A small amount (alpha) is added to the standard deviation to stop the
        normalization from exploding.

     @param alpha: a small value that will be added to the standard deviation
     @return returns true if the matrix was normalized, false otherwise
   */
  bool        znorm(const float alpha = 0.001);

  /**
     Performs the multiplication of the data by the scalar value.

     @return returns a new MatrixFloat with the results from the multiplcation
   */
  MatrixFloat multiple(const float value) const;

  /**
     Performs the multiplcation of this matrix (a) by the vector b.
     This will return a new vector (c): c = a * b

     @param b: the vector to multiple with this matrix
     @return a VectorFloat with the results from the multiplcation
   */
  VectorFloat multiple(const VectorFloat& b) const;

  /**
     Performs the multiplcation of this matrix (a) by the matrix b.
     This will return a new matrix (c): c = a * b

     @param b: the matrix to multiple with this matrix
     @return a new MatrixFloat with the results from the multiplcation
   */
  MatrixFloat multiple(const MatrixFloat& b) const;

  /**
     Performs the multiplcation of the matrix a by the matrix b, directly
        storing the new data in the this matrix instance.
     This will resize the current matrix if needed.
     This makes this matrix c and gives: c = a * b, or if the aTransposed value
        is true: c = a' * b

     @param a: the matrix to multiple with b
     @param b: the matrix to multiple with a
     @param aTranspose: a flag to indicate if matrix a should be transposed
     @return true if the operation was completed successfully, false otherwise
   */
  bool        multiple(const MatrixFloat& a,
                       const MatrixFloat& b,
                       const bool         aTranspose = false);

  /**
     Adds the input matrix data (b) to this matrix (a), giving: a = a + b.
     This rows and cols of b must match that of this matrix.

     @param a: the matrix to multiple with b
     @param b: the matrix to multiple with a
     @param aTranspose: a flag to indicate if matrix a should be transposed
     @return true if the operation was completed successfully, false otherwise
   */
  bool add(const MatrixFloat& b);

  /**
     Adds the input matrix data (a) to the input matrix (b), storing the data in
        this matrix (c) giving: c = a + b.
     The rows and cols in a and b must match.
     This will resize the current matrix if needed.

     @param a: the matrix to add with b
     @param b: the matrix to add with a
     @return true if the operation was completed successfully, false otherwise
   */
  bool add(const MatrixFloat& a,
           const MatrixFloat& b);

  /**
     Subtracts the input matrix data (b) from this matrix (a), giving: a = a -
        b.
     This rows and cols of b must match that of this matrix.

     @param a: the matrix to subtract from this instance
     @return true if the operation was completed successfully, false otherwise
   */
  bool subtract(const MatrixFloat& b);

  /**
     Subtracts the input matrix data (b) from this matrix (a), giving (c): c = a
        - b.
     This rows and cols of b must match that of this matrix.
     This will resize the current matrix if needed.

     @param a: the matrix to subtract with b
     @param b: the matrix to subtract from a
     @return true if the operation was completed successfully, false otherwise
   */
  bool subtract(const MatrixFloat& a,
                const MatrixFloat& b);

  /**
     Gets the ranges min value throughout the entire matrix.

     @return a Float value containing the minimum matrix value
   */
  float         getMinValue() const;

  /**
     Gets the ranges max value throughout the entire matrix.

     @return a Float value containing the maximum matrix value
   */
  float         getMaxValue() const;

  /**
     Gets the mean of each column in the matrix and returns this as a
        VectorFloat.

     @return a VectorFloat with the mean of each column
   */
  VectorFloat   getMean() const;

  /**
     Gets the standard deviation of each column in the matrix and returns this
        as a VectorFloat.

     @return a VectorFloat with the standard deviation of each column
   */
  VectorFloat   getStdDev() const;

  /**
     Gets the covariance matrix of this matrix and returns this as a
        MatrixFloat.

     @return a MatrixFloat with the covariance matrix of this matrix
   */
  MatrixFloat   getCovarianceMatrix() const;

  /**
     Gets the ranges (min and max values) of each column in the matrix.

     @return a vector with the ranges (min and max values) of each column in the
        matrix
   */
  Vector<MinMax>getRanges() const;

  /**
     Gets the trace of this matrix.

     @return the trace of this matrix as a Float
   */
  float         getTrace() const;

protected:

  float stringToFloat(const FString& value) {
    return FCString::Atof(*value);
  }
};
}
