#ifndef _TENSOR3_HPP_
#define _TENSOR3_HPP_

class Tensor3 {
public:
  Tensor3() { data_ = NULL; }
  Tensor3(int N) : N_(N) { data_ = new double[N * N * N]; }
  ~Tensor3() { if (data_ != NULL) free(data_); }

  // Copy constructor
  Tensor3(Tensor3& that) : Tensor3(that.dim()) {
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
  Tensor3(Tensor3&& that) : Tensor3() {
    swap(*this, that);
  }

  // copy assignment
  Tensor3& operator=(Tensor3 that) {
    swap(*this, that);
    return *this;
  }

  friend void swap(Tensor3& first, Tensor3& second) {
    using std::swap;
    swap(first.N_, second.N_);
    swap(first.data_, second.data_);
  }

  double Get(int i, int j, int k) const {
    return data_[k + j * N_ + i * N_ * N_];
  }

  void Set(int i, int j, int k, double val) {
    data_[k + j * N_ + i * N_ * N_] = val;
  }

  // Get T(:, j, k)
  std::vector<double> GetSlice1(int j, int k) const {
    std::vector<double> col(N_);
    for (int i = 0; i < N_; ++i) {
      col[i] = Get(i, j, k);
    }
    return col;
  }

  // Get T(i, j, :)
  std::vector<double> GetSlice3(int i, int j) const {
    std::vector<double> col(N_);
    for (int k = 0; k < N_; ++k) {
      col[k] = Get(i, j, k);
    }
    return col;
  }

  // Set T(:, j, k)
  void SetSlice1(int j, int k, const std::vector<double>& col) {
    assert(col.size() >= N_);
    for (int i = 0; i < N_; ++i) {
      Set(i, j, k, col[i]);
    }
  }

  // Set T(i, j, :)
  void SetSlice3(int i, int j, const std::vector<double>& col) {
    assert(col.size() >= N_);
    for (int k = 0; k < N_; ++k) {
      Set(i, j, k, col[k]);
    }
  }

  void SetGlobalValue(double val) {
    for (int i = 0; i < N_; ++i) {
      for (int j = 0; j < N_; ++j) {
	for (int k = 0; k < N_; ++k) {
	  Set(i, j, k, val);
	}
      }
    }
  }

  int dim() const { return N_; }

  const double& operator()(int i, int j, int k) const { 
    return data_[k + j * N_ + i * N_ * N_];
  }
  double& operator()(int i, int j, int k) {
    return data_[k + j * N_ + i * N_ * N_];
  }

private:
  double *data_;
  int N_;
};

class Matrix2 {
public:
  Matrix2() { data_ = NULL; }
  Matrix2(int N) : N_(N) { data_ = new double[N * N]; }
  ~Matrix2() { if (data_ != NULL) free(data_); }

  // Copy constructor
  Matrix2(Matrix2& that) : Matrix2(that.dim()) {
    int N = that.dim();
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
	Set(i, j, that(i, j));
      }
    }
  }

  // Move constructor
  Matrix2(Matrix2&& that) : Matrix2() {
    swap(*this, that);
  }

  // copy assignment
  Matrix2& operator=(Matrix2 that) {
    swap(*this, that);
    return *this;
  }

  friend void swap(Matrix2& first, Matrix2& second) {
    using std::swap;
    swap(first.N_, second.N_);
    swap(first.data_, second.data_);
  }

  double Get(int i, int j) const {
    return data_[j + i * N_];
  }

  void Set(int i, int j, double val) {
    data_[j + i * N_] = val;
  }

  void SetGlobalValue(double val) {
    for (int i = 0; i < N_; ++i) {
      for (int j = 0; j < N_; ++j) {
	Set(i, j, val);
      }
    }
  }

  // Get T(:, j)
  std::vector<double> GetColumn(int j) const {
    std::vector<double> col(N_);
    for (int i = 0; i < N_; ++i) {
      col[i] = Get(i, j);
    }
    return col;
  }

  // Set T(:, j)
  void SetColumn(int j, const std::vector<double>& col) {
    assert(col.size() >= N_);
    for (int i = 0; i < N_; ++i) {
      Set(i, j, col[i]);
    }
  }

  int dim() const { return N_; }

  const double& operator()(int i, int j) const { 
    return data_[j + i * N_];
  }
  double& operator()(int i, int j) {
    return data_[j + i * N_];
  }

private:
  double *data_;
  int N_;
};

#endif  // _TENSOR3_HPP_
