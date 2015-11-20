#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "math.h"

#include "tensor3.hpp"

static double fraction_init = 0.2;
static double fraction_train = 0.4;

template <typename T>
std::vector<double> ScaledVec(const std::vector<T>& vec,
			      double val) {
  std::vector<double> svec(vec.size());
  for (int i = 0; i < vec.size(); ++i) {
    svec[i] = static_cast<double>(vec[i]) * val;
  }
  return svec;
}

int StartingIndex(std::vector<int>& seq) {
  return floor((fraction_init) * seq.size());
}

int EndingIndex(std::vector<int>& seq) {
  return floor((fraction_init + fraction_train) * seq.size());
}

std::vector<int> InitializeHistory(std::vector<int>& seq,
				   int dimension) {
  std::vector<int> history(dimension, 1);
  for (int i = 0; i < StartingIndex(seq); ++i) {
    history[seq[i]] += 1;
  }
  return history;
}

int Sum(std::vector<int>& vec) {
  int sum = 0;
  for (int v : vec) {
    sum += v;
  }
  return sum;
}

Tensor3 Gradient(std::vector< std::vector<int> >& seqs,
		 Tensor3& P) {
  Tensor3 G(P.dim());
  G.SetGlobalValue(0.0);
  for (auto& seq : seqs) {
    std::vector<int> history = InitializeHistory(seq, P.dim());
    int total = Sum(history);
    for (int l = StartingIndex(seq); l < EndingIndex(seq); ++l) {
      int i = seq[l];
      int j = seq[l - 1];
      auto occupancy =
	ScaledVec(history, 1.0 / static_cast<double>(total));
      double sum = 0.0;
      for (int k = 0; k < P.dim(); ++k) {
	sum += occupancy[k] * P(i, j, k);
      }
      for (int k = 0; k < P.dim(); ++k) {
	G(i, j, k) = G.Get(i, j, k) + occupancy[k] / sum;
      }
      history[i] += 1;
      ++total;
    }
  }
  return G;
}

double LogLikelihoodTrain(std::vector< std::vector<int> >& seqs, Tensor3& P) {
  double ll = 0.0;
  for (auto& seq : seqs) {
    std::vector<int> history = InitializeHistory(seq, P.dim());
    int total = Sum(history);
    for (int l = StartingIndex(seq); l < EndingIndex(seq); ++l) {
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

double SpaceyRMSE(std::vector< std::vector<int> >& seqs, Tensor3& P) {
  double ll = 0.0;
  double err = 0.0;
  int num = 0;
  for (auto& seq : seqs) {
    std::vector<int> history(P.dim(), 1);
    for (int l = 0; l < EndingIndex(seq); ++l) {
      history[seq[l]] += 1;
    }
    int total = Sum(history);
    for (int l = EndingIndex(seq); l < seq.size(); ++l) {
      int i = seq[l];
      int j = seq[l - 1];
      auto occupancy =
	ScaledVec(history, 1.0 / static_cast<double>(total));
      double sum = 0.0;
      for (int k = 0; k < occupancy.size(); ++k) {
	sum += occupancy[k] * P(i, j, k);
      }
      double val = 1 - sum;
      err += val * val;
      ++num;
    }
  }
  return sqrt(err / num);
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
    for (int l = StartingIndex(seq); l < EndingIndex(seq); ++l) {
      int k = seq[l - 2];
      int j = seq[l - 1];
      int i = seq[l];
      X(i, j, k) = X.Get(i, j, k) + 1;
    }
  }
  NormalizeStochastic(X);
  return X;
}

void GradientDescent(std::vector< std::vector<int> >& seqs) {
  Tensor3 X = InitializeEmpirical(seqs);
  double curr_ll = LogLikelihoodTrain(seqs, X);
  double generalization_error = SpaceyRMSE(seqs, X);
  std::cerr << "Starting log likelihood: " << curr_ll << std::endl;
  std::cerr << "Generalization RMSE: " << generalization_error << std::endl;
  double last_ll = curr_ll;

  int niter = 1000;
  int iter = 0;

  double step_size = 1.0;
  while (iter < niter && step_size > 1e-16) {
    Tensor3 Grad = Gradient(seqs, X);
    Tensor3 Y = UpdateProbs(X, Grad, step_size);
    NormalizeStochastic(Y);
    double test_ll = LogLikelihoodTrain(seqs, Y);
    double gen_rmse = SpaceyRMSE(seqs, Y);
    if (test_ll > curr_ll) {
      curr_ll = test_ll;
      X = Y;
      std::cerr << curr_ll << " " << step_size << " (" << gen_rmse << ")" << std::endl;
      step_size = step_size * 0.9;
    } else {
      step_size = step_size * 0.1;
    }
  }
}

double SecondOrderRMSE(std::vector< std::vector<int> >& seqs) {
  Tensor3 P = InitializeEmpirical(seqs);
  double err = 0.0;
  int num = 0;
  for (auto& seq : seqs) {
    for (int l = EndingIndex(seq); l < seq.size(); ++l) {
      int k = seq[l - 2];
      int j = seq[l - 1];
      int i = seq[l];
      double val = 1 - P(i, j, k);
      err += val * val;
      ++num;
    }
  }
  return sqrt(err / num);
}

double FirstOrderRMSE(std::vector< std::vector<int> >& seqs) {
  // Fill in empirical transitions
  int dim = MaximumIndex(seqs) + 1;
  Matrix2 X(dim);
  X.SetGlobalValue(0);
  for (auto& seq : seqs) {
    for (int l = StartingIndex(seq); l < EndingIndex(seq); ++l) {
      int j = seq[l - 1];
      int i = seq[l];
      X(i, j) = X.Get(i, j) + 1;
    }
  }

  // Normalize to stochastic
  for (int j = 0; j < dim; ++j) {
    std::vector<double> col = X.GetColumn(j);
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
    X.SetColumn(j, col);
  }

  // Compute Log Likelihood
  double err = 0;
  int num = 0;
  for (auto& seq : seqs) {
    for (int l = EndingIndex(seq); l < seq.size(); ++l) {
      int j = seq[l - 1];
      int i = seq[l];
      double val = (1 - X(i, j));
      err += val * val;
      ++num;
    }
  }
  return sqrt(err / num);
}

int main(int argc, char **argv) {
  std::vector< std::vector<int> > seqs;
  ReadSequences(argv[1], seqs);
  std::cerr << seqs.size() << " sequences." << std::endl;
  std::cerr << "First order RMSE: "
	    << FirstOrderRMSE(seqs) << std::endl
	    << "Second order RMSE: "
	    << SecondOrderRMSE(seqs) << std::endl;
  GradientDescent(seqs);
}
