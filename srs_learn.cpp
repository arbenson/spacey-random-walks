#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "math.h"

#include "tensor3.hpp"


template <typename T>
std::vector<double> ScaledVec(const std::vector<T>& vec,
			      double val) {
  std::vector<double> svec(vec.size());
  for (int i = 0; i < vec.size(); ++i) {
    svec[i] = static_cast<double>(vec[i]) * val;
  }
  return svec;
}

Tensor3 Gradient(std::vector< std::vector<int> >& seqs,
		 Tensor3& P) {
  Tensor3 G(P.dim());
  G.SetGlobalValue(0.0);
  for (auto& seq : seqs) {
    std::vector<int> history(P.dim(), 1);
    int total = P.dim();
    for (int l = 1; l < seq.size(); ++l) {
      int i = seq[l];
      int j = seq[l - 1];
      auto occupancy =
	ScaledVec(history, 1.0 / static_cast<double>(total));
      double sum = 0.0;
      for (int k = 0; k < P.dim(); ++k) {
	sum += occupancy[k] * P(i, j, k);
      }
      for (int k = 0; k < P.dim(); ++k) {
	G(i, j, k) = G(i, j, k) + occupancy[k] / sum;
      }
      history[i] += 1;
      ++total;
    }
  }
  return G;
}

double LogLikelihood(std::vector< std::vector<int> >& seqs,
		     Tensor3& P) {
  double ll = 0.0;
  for (auto& seq : seqs) {
    std::vector<int> history(P.dim(), 1);
    int total = P.dim();
    for (int l = 1; l < seq.size(); ++l) {
      int i = seq[l];
      int j = seq[l - 1];
      auto occupancy =
	ScaledVec(history, 1.0 / static_cast<double>(total));
      double sum = 0.0;
      for (int k = 0; k < occupancy.size(); ++k) {
	sum += occupancy[k] * P(i, j, k);
      }
      ll += log(sum);
      history[i] += 1;
      ++total;
    }
  }
  return ll;
}

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

int MaximumIndex(std::vector< std::vector<int> >& seqs) {
  int max_ind = 0;
  for (auto& seq : seqs) {
    for (int val : seq) {
      max_ind = std::max(max_ind, val);
    }
  }
  return max_ind;
}

Tensor3 UpdateProbs(Tensor3& X, Tensor3& G, double step_size) {
  int N = X.dim();
  Tensor3 Y(N);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < N; ++k) {
	double xijk = X(i, j, k);
	double gijk = G(i, j, k);
	double val = xijk + step_size * gijk;
	// This tolerance could be set to something else
	Y(i, j, k) = std::max(1e-12, std::min(val, 1 - 1e-12));
      }
    }
  }
  return Y;
}

Tensor3 InitializeEmpirical(std::vector< std::vector<int> >& seqs) {
  int dim = MaximumIndex(seqs) + 1;
  Tensor3 X(dim);
  X.SetGlobalValue(0);
  for (auto& seq : seqs) {
    for (int l = 2; l < seq.size(); ++l) {
      int k = seq[l - 2];
      int j = seq[l - 1];
      int i = seq[l];
      X(i, j, k) = X(i, j, k) + 1;
    }
  }
  NormalizeStochastic(X);
  return X;
}

void GradientDescent(std::vector< std::vector<int> >& seqs) {
  Tensor3 X = InitializeEmpirical(seqs);
  double curr_ll = LogLikelihood(seqs, X);
  std::cerr << "Starting log likelihood: " << curr_ll << std::endl;
  double last_ll = curr_ll;

  int niter = 1000;
  int iter = 0;

  double step_size = 1.0;
  while (iter < niter && step_size > 1e-16) {
    Tensor3 Grad = Gradient(seqs, X);
    Tensor3 Y = UpdateProbs(X, Grad, step_size);
    NormalizeStochastic(Y);
    double test_ll = LogLikelihood(seqs, Y);
    if (test_ll > curr_ll) {
      curr_ll = test_ll;
      X = Y;
      std::cerr << curr_ll << " " << step_size << std::endl;
      step_size = step_size * 0.9;
    } else {
      step_size = step_size * 0.1;
    }
  }
}

int main(int argc, char **argv) {
  std::vector< std::vector<int> > seqs;
  ReadSequences(argv[1], seqs);
  std::cerr << seqs.size() << " sequences." << std::endl;
  GradientDescent(seqs);
}
