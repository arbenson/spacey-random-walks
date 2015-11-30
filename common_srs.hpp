#ifndef _COMMON_SRS_HPP_
#define _COMMON_SRS_HPP_

#include <cmath>
#include <vector>

#include "tensor3.hpp"

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

// Maximum index used for all states (starts at 0).
int MaximumIndex(const std::vector< std::vector<int> >& seqs);

// Minimum l2 projection onto the simplex.
std::vector<double> SimplexProjection(const std::vector<double>& vec);

// Normalize columns of a transition probability tensor to be stochastic.
void NormalizeStochastic(Tensor3& P);

// Use the minimum l2 projection to project each column of the tensor onto the
// simplex.
void ProjectColumnsOntoSimplex(Tensor3& Y);

// Sample from a discrete probability distribution.
int Choice(const std::vector<double>& probs);

// || vec(P1) - vec(P2) ||_1
double L1Diff(const Tensor3& P1, const Tensor3& P2);

// || v1 - v2 ||_1
double L1Diff(const std::vector<double>& v1, const std::vector<double>& v2);

#endif  // _COMMON_SRS_HPP_
