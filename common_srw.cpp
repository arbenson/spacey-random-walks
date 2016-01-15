#include "common_srw.hpp"

#include <cassert>
#include <iostream>
#include <fstream>
#include <random>
#include <sstream>
#include <vector>

#include "hypermatrix.hpp"

bool InTopK(const std::vector<double>& vec, int index, int K) {
  std::vector< std::pair<double, int> > ind_vec(vec.size());
  for (int i = 0; i < vec.size(); ++i) {
    ind_vec[i] = std::pair<double, int>(-vec[i], i);
  }
  std::sort(ind_vec.begin(), ind_vec.end());
  for (int i = 0; i < K; ++i) {
    if (ind_vec[i].second == index) {
      return true;
    }
  }
  return false;
}

int MaximumIndex(const std::vector< std::vector<int> >& seqs) {
  int max_ind = 0;
  for (auto& seq : seqs) {
    for (int val : seq) {
      max_ind = std::max(max_ind, val);
    }
  }
  return max_ind;
}

std::vector<double> SimplexProjection(const std::vector<double>& vec) {
  std::vector<double> mu = vec;
  std::sort(mu.begin(), mu.end(), std::greater<double>());
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

void ProjectColumnsOntoSimplex(DblCubeHypermatrix& Y) {
  int dim = Y.dim();
  for (int j = 0; j < dim; ++j) {
    for (int k = 0; k < dim; ++k) {
      auto col = Y.GetSlice1(j, k);
      Y.SetSlice1(j, k, SimplexProjection(col));
    }
  }
}

void NormalizeStochastic(DblCubeHypermatrix& P) {
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

double L1Diff(const DblCubeHypermatrix& P1, const DblCubeHypermatrix& P2) {
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

double L1Diff(const std::vector<double>& v1, const std::vector<double>& v2) {
  double diff = 0.0;
  assert(v1.size() == v2.size());
  for (int i = 0; i < v1.size(); ++i) {
    diff += std::abs(v1[i] - v2[i]);
  }
  return diff;
}

std::vector<double> HypermatrixApply(const DblCubeHypermatrix& P,
				     const std::vector<double>& x) {
  std::vector<double> y(x.size(), 0.0);
  int dim = P.dim();
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      for (int k = 0; k < dim; ++k) {
        // P(i, j, k) is column (j, k) and row (i, j)
        y[i] += P.Get(i, j, k) * x[j] * x[k];
      }
    }
  }
  return y;
}

std::vector<double> Apply(const DblCubeHypermatrix& P,
			  const std::vector<double>& x) {
  std::vector<double> y(x.size(), 0.0);
  int dim = P.dim();
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      for (int k = 0; k < dim; ++k) {
        // P(i, j, k) is column (j, k) and row (i, j)
        y[i * dim + j] += P(i, j, k) * x[j * dim + k];
      }
    }
  }
  return y;
}


std::vector<double> Stationary(const DblCubeHypermatrix& P) {
  int dim = P.dim();
  std::vector<double> x(dim * dim, 1.0 / (dim * dim));
  int max_iter = 1000;
  double tol = 1e-12;
  for (int iter = 0; iter < max_iter; ++iter) {
    std::vector<double> x_next = Apply(P, x);
    // Check the difference
    double diff = L1Diff(x_next, x);
    x = x_next;
    x = Normalized(x);
    // Stop if difference is small enough
    if (diff < tol) { break; }
  }
  return x;
}

std::vector<double> StationaryMarginals(const DblCubeHypermatrix& P) {
  std::vector<double> st = Stationary(P);
  int dim = P.dim();
  std::vector<double> marginals(dim, 0.0);
  for (int i = 0; i < st.size(); ++i) {
    int marginal_ind = (i % dim);
    marginals[marginal_ind] += st[i];
  }
  return marginals;
}

std::vector<double> SpaceyStationary(const DblCubeHypermatrix& P,
				     int max_iter=1000, double gamma=0.01,
				     double tol=1e-12) {
  int dim = P.dim();
  std::vector<double> x(dim, 1.0 / dim);
  for (int iter = 0; iter < max_iter; ++iter) {
    std::vector<double> x_next = HypermatrixApply(P, x);
    // Check the difference
    double diff = L1Diff(x_next, x);
    for (int j = 0; j < x.size(); ++j) {
      x[j] = (1.0 - gamma) * x_next[j] + gamma * x[j];
    }
    x = Normalized(x);
    // Stop if difference is small enough
    if (diff < tol) { break; }
  }
  return x;
}

void ReadSequences(std::string filename,
                   std::vector< std::vector<int> >& seqs) {
  std::string line;
  std::ifstream infile(filename);
  while (std::getline(infile, line)) {
    int loc;
    char delim;
    std::vector<int> seq;
    std::istringstream iss(line);
    while (iss >> loc) {
      seq.push_back(loc);
      iss >> delim;
    }
    seqs.push_back(seq);
  }
}

void WriteHypermatrix(const DblCubeHypermatrix& P, const std::string& outfile) {
  std::ofstream out;
  out.open(outfile);
  int dim = P.dim();
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      for (int k = 0; k < dim; ++k) {
	out << i << " "
	    << j << " "
	    << k << " "
	    << P(i, j, k) << std::endl;
      }
    }
  }
  out.close();
}
