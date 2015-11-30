#include "common_srs.hpp"
#include <iostream>

int MaximumIndex(std::vector< std::vector<int> >& seqs) {
  int max_ind = 0;
  for (auto& seq : seqs) {
    for (int val : seq) {
      max_ind = std::max(max_ind, val);
    }
  }
  return max_ind;
}

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
  assert(rho + 1.0 > 0.0);
  double theta = (csum - 1.0) / (rho + 1.0);

  std::vector<double> ret = vec;
  for (int i = 0; i < ret.size(); ++i) {
    ret[i] = std::max(vec[i] - theta, 0.0);
  }
  return ret;
}

void Project(Tensor3& Y) {
  int dim = Y.dim();
  for (int j = 0; j < dim; ++j) {
    for (int k = 0; k < dim; ++k) {
      auto col = Y.GetSlice1(j, k);
      Y.SetSlice1(j, k, EuclideanProjectSimplex(col));
    }
  }
}

void NormalizeStochastic(Tensor3& P) {
  int dim = P.dim();
  for (int k = 0; k < dim; ++k) {
    for (int j = 0; j < dim; ++j) {
      std::vector<double> col = P.GetSlice1(j, k);
      double sum = Sum(col);
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

int Choice(const std::vector<double>& probs) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0, 1);
  double val = dis(gen);

  double csum = 0.0;
  for (int i = 0; i < probs.size(); ++i) {
    csum += probs[i];
    if (val <= csum) {
      return i;
    }
  }
  std::cerr << "WARNING: Probability vector did not sum to 1."
	    << std::endl;
  return probs.size() - 1;
}

double L1Diff(const Tensor3& P1, const Tensor3& P2) {
  double diff = 0.0;
  int dimension = P1.dim();
  assert(dimension == P2.dim());
  for (int i = 0; i < dimension; ++i) {
    for (int j = 0; j < dimension; ++j) {
      for (int k = 0; k < dimension; ++k) {
        diff += std::abs(P1(i, j, k) - P2(i, j, k));
      }
    }
  }
  return diff;
}

double L1Diff(std::vector<double>& v1, std::vector<double>& v2) {
  double diff = 0.0;
  assert(v1.size() == v2.size());
  for (int i = 0; i < v1.size(); ++i) {
    diff += std::abs(v1[i] - v2[i]);
  }
  return diff;
}
