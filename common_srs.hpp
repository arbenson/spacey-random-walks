#ifndef _COMMON_SRS_HPP_
#define _COMMON_SRS_HPP_

#include "tensor3.hpp"

int MaximumIndex(std::vector< std::vector<int> >& seqs) {
  int max_ind = 0;
  for (auto& seq : seqs) {
    for (int val : seq) {
      max_ind = std::max(max_ind, val);
    }
  }
  return max_ind;
}

// Minimum l2 projection onto the simplex.
std::vector<double> EuclideanProjectSimplex(const std::vector<double>& vec) {
  std::vector<double> mu = vec;
  std::sort(mu.begin(), mu.end(), std::greater<int>());
  // Get cumulative sum
  double csum = 0.0;
  int rho = 0;
  for (int j = 0; j < mu.size(); ++j) {
    csum += mu[j];
    if (mu[j] - (csum - 1.0) / (j + 1) > 0) {
      rho = j;
    }
  }

  // Get the lagrange multiplier
  csum = 0;
  for (int j = 0; j <= rho; ++j) {
    csum += mu[j];
  }
  double theta = (csum - 1.0) / (rho + 1.0);

  std::vector<double> ret = vec;
  for (int i = 0; i < ret.size(); ++i) {
    ret[i] = std::max(vec[i] - theta, 0.0);
  }
  return ret;
}

// Normalize columns of a transition probability tensor to be stochastic.
void NormalizeStochastic(Tensor3& P) {
  int dim = P.dim();
  for (int k = 0; k < dim; ++k) {
    for (int j = 0; j < dim; ++j) {
      std::vector<double> col = P.GetSlice1(j, k);
      double sum = 0.0;
      for (int i = 0; i < col.size(); ++i) {
	sum += col[i];
      }
      if (sum == 0.0) {
	col = std::vector<double>(col.size(), 1.0 / col.size());
      } else {
	for (int i = 0; i < col.size(); ++i) {
	  col[i] /= sum;
	}
      }
      P.SetSlice1(j, k, col);
    }
  }
}

// Use the minimum l2 projection to project each column of the
// tensor onto the simplex.
void Project(Tensor3& Y) {
  int dim = Y.dim();
  for (int j = 0; j < dim; ++j) {
    for (int k = 0; k < dim; ++k) {
      auto col = Y.GetSlice1(j, k);
      Y.SetSlice1(j, k, EuclideanProjectSimplex(col));
    }
  }
}

template <typename T>
T Sum(const std::vector<T>& vec) {
  T sum = 0;
  for (T v : vec) {
    sum += v;
  }
  return sum;
}

template <typename T>
T AbsSum(const std::vector<T>& vec) {
  T sum = 0;
  for (T v : vec) {
    sum += std::abs(v);
  }
  return sum;
}

template <typename T>
std::vector<double> Normalized(const std::vector<T>& vec) {
  std::vector<double> nvec(vec.size());
  double sum = static_cast<double>(AbsSum(vec));
  for (int i = 0; i < nvec.size(); ++i) {
    nvec[i] = static_cast<double>(vec[i]) / sum;
  }
  return nvec;
}

#endif  // _COMMON_SRS_HPP_
