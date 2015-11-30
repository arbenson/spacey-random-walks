#ifndef _TENSOR3_HPP_
#define _TENSOR3_HPP_

#include <cassert>

// Simple wrapper around an N x N x N tensor
template <typename T>
class CubeTensor {
public:
  CubeTensor() { data_ = NULL; }
  CubeTensor(int N) : N_(N) { data_ = new T[N * N * N]; }
  ~CubeTensor() { if (data_ != NULL) free(data_); }

  // Copy constructor
  CubeTensor(CubeTensor& that) : CubeTensor(that.dim()) {
    int N = that.dim();
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
	for (int k = 0; k < N; ++k) {
	  Set(i, j, k, that(i, j, k));
	}
      }
    }
  }

  // Move constructor
  CubeTensor(CubeTensor&& that) : CubeTensor() {
    swap(*this, that);
  }

  // copy assignment
  CubeTensor& operator=(CubeTensor that) {
    swap(*this, that);
    return *this;
  }

  friend void swap(CubeTensor& first, CubeTensor& second) {
    using std::swap;
    swap(first.N_, second.N_);
    swap(first.data_, second.data_);
  }

  T Get(int i, int j, int k) const {
    return data_[k + j * N_ + i * N_ * N_];
  }

  void Set(int i, int j, int k, T val) {
    data_[k + j * N_ + i * N_ * N_] = val;
  }

  // Get T(:, j, k)
  std::vector<T> GetSlice1(int j, int k) const {
    std::vector<T> col(N_);
    for (int i = 0; i < N_; ++i) {
      col[i] = Get(i, j, k);
    }
    return col;
  }

  // Get T(i, j, :)
  std::vector<T> GetSlice3(int i, int j) const {
    std::vector<T> col(N_);
    for (int k = 0; k < N_; ++k) {
      col[k] = Get(i, j, k);
    }
    return col;
  }

  // Set T(:, j, k)
  void SetSlice1(int j, int k, const std::vector<T>& col) {
    assert(col.size() >= N_);
    for (int i = 0; i < N_; ++i) {
      Set(i, j, k, col[i]);
    }
  }

  // Set T(i, j, :)
  void SetSlice3(int i, int j, const std::vector<T>& col) {
    assert(col.size() >= N_);
    for (int k = 0; k < N_; ++k) {
      Set(i, j, k, col[k]);
    }
  }

  void SetGlobalValue(T val) {
    for (int i = 0; i < N_; ++i) {
      for (int j = 0; j < N_; ++j) {
	for (int k = 0; k < N_; ++k) {
	  Set(i, j, k, val);
	}
      }
    }
  }

  int dim() const { return N_; }

  const T& operator()(int i, int j, int k) const { 
    return data_[k + j * N_ + i * N_ * N_];
  }
  T& operator()(int i, int j, int k) {
    return data_[k + j * N_ + i * N_ * N_];
  }

private:
  T *data_;
  int N_;
};


// Simple wrapper around an N x N tensor
template <typename T>
class SquareMatrix {
public:
  SquareMatrix() { data_ = NULL; }
  SquareMatrix(int N) : N_(N) { data_ = new T[N * N]; }
  ~SquareMatrix() { if (data_ != NULL) free(data_); }

  // Copy constructor
  SquareMatrix(SquareMatrix& that) : SquareMatrix(that.dim()) {
    int N = that.dim();
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
	Set(i, j, that(i, j));
      }
    }
  }

  // Move constructor
  SquareMatrix(SquareMatrix&& that) : SquareMatrix() {
    swap(*this, that);
  }

  // copy assignment
  SquareMatrix& operator=(SquareMatrix that) {
    swap(*this, that);
    return *this;
  }

  friend void swap(SquareMatrix& first, SquareMatrix& second) {
    using std::swap;
    swap(first.N_, second.N_);
    swap(first.data_, second.data_);
  }

  T Get(int i, int j) const {
    return data_[j + i * N_];
  }

  void Set(int i, int j, T val) {
    data_[j + i * N_] = val;
  }

  void SetGlobalValue(T val) {
    for (int i = 0; i < N_; ++i) {
      for (int j = 0; j < N_; ++j) {
	Set(i, j, val);
      }
    }
  }

  // Get T(:, j)
  std::vector<T> GetColumn(int j) const {
    std::vector<T> col(N_);
    for (int i = 0; i < N_; ++i) {
      col[i] = Get(i, j);
    }
    return col;
  }

  // Set T(:, j)
  void SetColumn(int j, const std::vector<T>& col) {
    assert(col.size() >= N_);
    for (int i = 0; i < N_; ++i) {
      Set(i, j, col[i]);
    }
  }

  int dim() const { return N_; }

  const T& operator()(int i, int j) const { return data_[j + i * N_]; }
  T& operator()(int i, int j) { return data_[j + i * N_]; }

private:
  T *data_;
  int N_;
};

typedef SquareMatrix<double> Matrix2;
typedef CubeTensor<double> Tensor3;

#endif  // _TENSOR3_HPP_
