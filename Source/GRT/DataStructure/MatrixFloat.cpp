#include "../GRT.h"
#include "MatrixFloat.h"

namespace GRT {
  MatrixFloat::MatrixFloat() {
    this->dataPtr = NULL;
    this->rowPtr = NULL;
    this->rows = 0;
    this->cols = 0;
  }

  MatrixFloat::MatrixFloat(const uint32 rows, const uint32 cols) {
    this->dataPtr = NULL;
    this->rowPtr = NULL;
    this->rows = 0;
    this->cols = 0;
    if (rows > 0 && cols > 0) {
      resize(rows, cols);
    }
  }

  MatrixFloat::MatrixFloat(const MatrixFloat &rhs) {
    this->dataPtr = NULL;
    this->rowPtr = NULL;
    this->rows = 0;
    this->cols = 0;
    this->copy(rhs);
  }

  MatrixFloat::MatrixFloat(const Matrix< float > &rhs) {
    this->dataPtr = NULL;
    this->rowPtr = NULL;
    this->rows = 0;
    this->cols = 0;
    this->copy(rhs);
  }

  MatrixFloat::MatrixFloat(const Vector< VectorFloat > &rhs) {
    this->dataPtr = NULL;
    this->rowPtr = NULL;
    this->rows = 0;
    this->cols = 0;

    if (rhs.size() == 0) return;

    uint32 M = rhs.getSize();
    uint32 N = (uint32)rhs[0].getSize();
    resize(M, N);

    for (uint32 i = 0; i<M; i++) {
      if (rhs[i].size() != N) {
        clear();
        return;
      }
      for (uint32 j = 0; j<N; j++) {
        dataPtr[i*cols + j] = rhs[i][j];
      }
    }
  }

  MatrixFloat::~MatrixFloat() {
    clear();
  }

  MatrixFloat& MatrixFloat::operator=(const MatrixFloat &rhs) {
    if (this != &rhs) {
      this->clear();
      this->copy(rhs);
    }
    return *this;
  }

  MatrixFloat& MatrixFloat::operator=(const Matrix< float > &rhs) {
    if (this != &rhs) {
      this->clear();
      this->copy(rhs);
    }
    return *this;
  }

  MatrixFloat& MatrixFloat::operator=(const Vector< VectorFloat > &rhs) {

    clear();

    if (rhs.size() == 0) return *this;

    uint32 M = rhs.getSize();
    uint32 N = (uint32)rhs[0].getSize();
    resize(M, N);

    for (uint32 i = 0; i<M; i++) {
      if (rhs[i].size() != N) {
        clear();
        return *this;
      }
      for (uint32 j = 0; j<N; j++) {
        dataPtr[i*cols + j] = rhs[i][j];
      }
    }

    return *this;
  }

  bool MatrixFloat::transpose() {

    if (dataPtr == NULL) return false;

    MatrixFloat temp(cols, rows);
    for (uint32 i = 0; i<rows; i++) {
      for (uint32 j = 0; j<cols; j++) {
        temp[j][i] = dataPtr[i*cols + j];
      }
    }

    *this = temp;

    return true;
  }

  bool MatrixFloat::scale(const float minTarget, const float maxTarget) {

    if (dataPtr == NULL) return false;

    Vector< MinMax > ranges = getRanges();

    return scale(ranges, minTarget, maxTarget);
  }

  bool MatrixFloat::scale(const Vector< MinMax > &ranges, const float minTarget, const float maxTarget) {
    if (dataPtr == NULL) return false;

    if (ranges.size() != cols) {
      return false;
    }

    uint32 i, j = 0;
    for (i = 0; i<rows; i++) {
      for (j = 0; j<cols; j++) {
        dataPtr[i*cols + j] = grt_scale(dataPtr[i*cols + j], ranges[j].minValue, ranges[j].maxValue, minTarget, maxTarget);
      }
    }
    return true;
  }

  bool MatrixFloat::znorm(const float alpha) {
    if (dataPtr == NULL) return false;

    uint32 i, j = 0;
    float mean, std = 0;
    for (i = 0; i<rows; i++) {
      mean = 0;
      std = 0;

      //Compute the mean
      for (j = 0; j<cols; j++) {
        mean += dataPtr[i*cols + j];
      }
      mean /= cols;

      //Compute the std dev
      for (j = 0; j<cols; j++) {
        std += (dataPtr[i*cols + j] - mean)*(dataPtr[i*cols + j] - mean);
      }
      std /= cols;
      std = sqrt(std + alpha);

      //Normalize the row
      for (j = 0; j<cols; j++) {
        dataPtr[i*cols + j] = (dataPtr[i*cols + j] - mean) / std;
      }
    }

    return true;
  }

  MatrixFloat MatrixFloat::multiple(const float value) const {

    if (dataPtr == NULL) return MatrixFloat();

    MatrixFloat d(rows, cols);
    float *d_p = &(d[0][0]);

    uint32 i = 0;
    for (i = 0; i<rows*cols; i++) {
      d_p[i] = dataPtr[i] * value;
    }

    return d;
  }

  VectorFloat MatrixFloat::multiple(const VectorFloat &b) const {

    const uint32 M = rows;
    const uint32 N = cols;
    const uint32 K = (uint32)b.size();

    if (N != K) {
      UE_LOG(GRTModule, Warning, TEXT("multiple(vector b) - The size of b %d does not match the number of columns in this matrix %d"), K, N);
      return VectorFloat();
    }

    VectorFloat c(M);
    const float *pb = &b[0];
    float *pc = &c[0];

    unsigned int i, j = 0;
    for (i = 0; i<rows; i++) {
      pc[i] = 0;
      for (j = 0; j<cols; j++) {
        pc[i] += dataPtr[i*cols + j] * pb[j];
      }
    }

    return c;
  }

  MatrixFloat MatrixFloat::multiple(const MatrixFloat &b) const {

    const uint32 M = rows;
    const uint32 N = cols;
    const uint32 K = b.getNumRows();
    const uint32 L = b.getNumCols();

    if (N != K) {
      UE_LOG(GRTModule, Warning, TEXT("multiple(vector b) - The size of b %d does not match the number of columns in this matrix %d"), K, N);
      return MatrixFloat();
    }

    MatrixFloat c(M, L);
    float **pb = b.getDataPointer();
    float **pc = c.getDataPointer();

    uint32 i, j, k = 0;
    for (i = 0; i<M; i++) {
      for (j = 0; j<L; j++) {
        pc[i][j] = 0;
        for (k = 0; k<K; k++) {
          pc[i][j] += dataPtr[i*cols + k] * pb[k][j];
        }
      }
    }

    return c;
  }

  bool MatrixFloat::multiple(const MatrixFloat &a, const MatrixFloat &b, const bool aTranspose) {

    const uint32 M = !aTranspose ? a.getNumRows() : a.getNumCols();
    const uint32 N = !aTranspose ? a.getNumCols() : a.getNumRows();
    const uint32 K = b.getNumRows();
    const uint32 L = b.getNumCols();

    if (N != K) {
      UE_LOG(GRTModule, Error, TEXT("multiple(const MatrixFloat &a,const MatrixFloat &b,const bool aTranspose) - The number of rows in a %d does not match the number of columns in matrix b %d"), K, N);
      return false;
    }

    if (!resize(M, L)) {
      UE_LOG(GRTModule, Error, TEXT("multiple(const MatrixFloat &b,const MatrixFloat &c,const bool bTranspose) - Failed to resize matrix!"), K, N);
      return false;
    }

    uint32 i, j, k = 0;

    //Using direct pointers really helps speed up the computation time
    float **pa = a.getDataPointer();
    float **pb = b.getDataPointer();

    if (aTranspose) {

      for (j = 0; j<L; j++) {
        for (i = 0; i<M; i++) {
          dataPtr[i*cols + j] = 0;
          for (k = 0; k<K; k++) {
            dataPtr[i*cols + j] += pa[k][i] * pb[k][j];
          }
        }
      }

    }
    else {

      for (j = 0; j<L; j++) {
        for (i = 0; i<M; i++) {
          dataPtr[i*cols + j] = 0;
          for (k = 0; k<K; k++) {
            dataPtr[i*cols + j] += pa[i][k] * pb[k][j];
          }
        }
      }

    }

    return true;
  }

  bool MatrixFloat::add(const MatrixFloat &b) {

    if (b.getNumRows() != rows) {
      UE_LOG(GRTModule, Error, TEXT("add(const MatrixFloat &b) - Failed to add matrix! The rows do not match!"));
      return false;
    }

    if (b.getNumCols() != cols) {
      UE_LOG(GRTModule, Error, TEXT("add(const MatrixFloat &b) - Failed to add matrix! The cols do not match!"));
      return false;
    }

    uint32 i = 0;

    //Using direct pointers really helps speed up the computation time
    const float *p_b = &(b[0][0]);

    for (i = 0; i<rows*cols; i++) {
      dataPtr[i] += p_b[i];
    }

    return true;
  }

  bool MatrixFloat::add(const MatrixFloat &a, const MatrixFloat &b) {

    const uint32 M = a.getNumRows();
    const uint32 N = a.getNumCols();

    if (M != b.getNumRows()) {
      UE_LOG(GRTModule, Error, TEXT("add(const MatrixFloat &a,const MatrixFloat &b) - Failed to add matrix! The rows do not match!, a rows: %d b rows: %d"), M, (uint32)b.getNumRows());
      return false;
    }

    if (N != b.getNumCols()) {
      UE_LOG(GRTModule, Error, TEXT("add(const MatrixFloat &a,const MatrixFloat &b) - Failed to add matrix! The columns do not match!, a cols: %d b cols: %d"), N, (uint32)b.getNumCols());
      return false;
    }

    resize(M, N);

    uint32 i;

    //Using direct pointers really helps speed up the computation time
    float *pa = a.getData();
    float *pb = b.getData();

    const uint32 sizeMN = M*N;
    for (i = 0; i<sizeMN; i++) {
      dataPtr[i] = pa[i] + pb[i];
    }

    return true;
  }

  bool MatrixFloat::subtract(const MatrixFloat &b) {

    if (b.getNumRows() != rows) {
      UE_LOG(GRTModule, Error, TEXT("subtract(const MatrixFloat &b) - Failed to add matrix! The rows do not match! b rows: %d"), (uint32)b.getNumRows());
      return false;
    }

    if (b.getNumCols() != cols) {
      UE_LOG(GRTModule, Error, TEXT("subtract(const MatrixFloat &b) - Failed to add matrix! The cols do not match! b cols: %d"), (uint32)b.getNumCols());
      return false;
    }

    uint32 i;

    //Using direct pointers really helps speed up the computation time
    float *pb = b.getData();

    const uint32 sizeRC = rows*cols;
    for (i = 0; i<sizeRC; i++) {
      dataPtr[i] -= pb[i];
    }

    return true;
  }

  bool MatrixFloat::subtract(const MatrixFloat &a, const MatrixFloat &b) {

    const uint32 M = a.getNumRows();
    const uint32 N = a.getNumCols();

    if (M != b.getNumRows()) {
      UE_LOG(GRTModule, Error, TEXT("subtract(const MatrixFloat &a,const MatrixFloat &b) - Failed to add matrix! The rows do not match! a rows: %d b rows: %d"), M, (uint32)b.getNumRows());
      return false;
    }

    if (N != b.getNumCols()) {
      UE_LOG(GRTModule, Error, TEXT("subtract(const MatrixFloat &a,const MatrixFloat &b) - Failed to add matrix! The cols do not match! a cols: %d b cols: %d"), N, (uint32)b.getNumCols());
      return false;
    }

    resize(M, N);

    uint32 i, j;

    //Using direct pointers really helps speed up the computation time
    float **pa = a.getDataPointer();
    float **pb = b.getDataPointer();

    for (i = 0; i<M; i++) {
      for (j = 0; j<N; j++) {
        dataPtr[i*cols + j] = pa[i][j] - pb[i][j];
      }
    }

    return true;
  }

  float MatrixFloat::getMinValue() const {
    float minValue = std::numeric_limits<float>::max();
    for (uint32 i = 0; i<rows*cols; i++) {
      if (dataPtr[i] < minValue) minValue = dataPtr[i];
    }
    return minValue;
  }

  float MatrixFloat::getMaxValue() const {
    float maxValue = std::numeric_limits<float>::lowest();
    for (uint32 i = 0; i<rows*cols; i++) {
      if (dataPtr[i] > maxValue) maxValue = dataPtr[i];
    }
    return maxValue;
  }

  VectorFloat MatrixFloat::getMean() const {

    VectorFloat mean(cols);

    for (uint32 c = 0; c<cols; c++) {
      mean[c] = 0;
      for (uint32 r = 0; r<rows; r++) {
        mean[c] += dataPtr[r*cols + c];
      }
      mean[c] /= float(rows);
    }

    return mean;
  }

  VectorFloat MatrixFloat::getStdDev() const {

    VectorFloat mean = getMean();
    VectorFloat stdDev(cols, 0);

    for (uint32 j = 0; j<cols; j++) {
      for (uint32 i = 0; i<rows; i++) {
        stdDev[j] += (dataPtr[i*cols + j] - mean[j])*(dataPtr[i*cols + j] - mean[j]);
      }
      stdDev[j] = sqrt(stdDev[j] / float(rows - 1));
    }
    return stdDev;
  }

  MatrixFloat MatrixFloat::getCovarianceMatrix() const {

    Vector<float> mean = getMean();
    MatrixFloat covMatrix(cols, cols);

    for (uint32 j = 0; j<cols; j++) {
      for (uint32 k = 0; k<cols; k++) {
        covMatrix[j][k] = 0;
        for (uint32 i = 0; i<rows; i++) {
          covMatrix[j][k] += (dataPtr[i*cols + j] - mean[j]) * (dataPtr[i*cols + k] - mean[k]);
        }
        covMatrix[j][k] /= float(rows - 1);
      }
    }

    return covMatrix;
  }

  Vector< MinMax > MatrixFloat::getRanges() const {

    if (rows == 0) return Vector< MinMax >();

    Vector< MinMax > ranges(cols);
    for (uint32 i = 0; i<rows; i++) {
      for (uint32 j = 0; j<cols; j++) {
        ranges[j].updateMinMax(dataPtr[i*cols + j]);
      }
    }
    return ranges;
  }

  float MatrixFloat::getTrace() const {
    float t = 0;
    uint32 K = (rows < cols ? rows : cols);
    for (uint32 i = 0; i < K; i++) {
      t += dataPtr[i*cols + i];
    }
    return t;
  }
}