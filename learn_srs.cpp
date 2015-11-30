#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "math.h"

#include "common_srs.hpp"
#include "tensor3.hpp"

static double fraction_init = 0.5;
static double fraction_train = 0.4;

int StartingIndex(std::vector<int>& seq) {
  return floor((fraction_init) * seq.size());
}

int EndingIndex(std::vector<int>& seq) {
  return floor((fraction_init + fraction_train) * seq.size());
}

std::vector<int> InitializeHistory(std::vector<int> seq, int dimension) {
  std::vector<int> history(dimension, 1);
  for (int i = 0; i < StartingIndex(seq); ++i) {
    history[seq[i]] += 1;
  }
  return history;
}

Tensor3 Gradient(std::vector< std::vector<int> >& seqs,
		 Tensor3& P) {
  Tensor3 G(P.dim());
  G.SetGlobalValue(0.0);
  for (auto& seq : seqs) {
    std::vector<int> history = InitializeHistory(seq, P.dim());
    for (int l = StartingIndex(seq); l < EndingIndex(seq); ++l) {
      auto occupancy = Normalized(history);
      int i = seq[l];
      int j = seq[l - 1];
      double sum = 0.0;
      for (int k = 0; k < P.dim(); ++k) {
	sum += occupancy[k] * P(i, j, k);
      }
      assert (sum > 0.0);
      for (int k = 0; k < P.dim(); ++k) {
	G(i, j, k) = G.Get(i, j, k) + occupancy[k] / sum;
      }
      history[i] += 1;
    }
  }
  return G;
}

double LogLikelihoodTrain(std::vector< std::vector<int> >& seqs, Tensor3& P) {
  double ll = 0.0;
  for (auto& seq : seqs) {
    std::vector<int> history = InitializeHistory(seq, P.dim());
    for (int l = StartingIndex(seq); l < EndingIndex(seq); ++l) {
      auto occupancy = Normalized(history);
      int i = seq[l];
      int j = seq[l - 1];
      double sum = 0.0;
      for (int k = 0; k < occupancy.size(); ++k) {
	sum += occupancy[k] * P(i, j, k);
      }
      ll += log(sum);
      history[i] += 1;
    }
  }
  return ll;
}

double SpaceyRMSE(std::vector< std::vector<int> >& seqs, Tensor3& P) {
  double ll = 0.0;
  double err = 0.0;
  int num = 0;
  for (auto& seq : seqs) {
    // std::vector<int> history = InitializeHistory(seq, P.dim());
    std::vector<int> history(P.dim(), 1);
    for (int i = 0; i < EndingIndex(seq); ++i) {
      history[seq[i]] += 1;
    }
    for (int l = EndingIndex(seq); l < seq.size(); ++l) {
      auto occupancy = Normalized(history);
      int i = seq[l];
      int j = seq[l - 1];
      double sum = 0.0;
      for (int k = 0; k < occupancy.size(); ++k) {
	sum += occupancy[k] * P(i, j, k);
      }
      double val = 1 - sum;
      err += val * val;
      ++num;
      history[i] += 1;
    }
  }
  return sqrt(err / num);
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

Tensor3 UpdateAndProject(Tensor3& X, Tensor3& G, double step_size) {
  int N = X.dim();
  Tensor3 Y(N);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < N; ++k) {
	double xijk = X(i, j, k);
	double gijk = G(i, j, k);
	Y(i, j, k) = xijk + step_size * gijk;
      }
    }
  }
  Project(Y);
  return Y;
}

Tensor3 EmpiricalSecondOrder(std::vector< std::vector<int> >& seqs,
			     bool spacey_correction) {
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

  // We need to check if there exists a (j, i) for which
  // P((k, j) --> (j, i)) is zero for all k
  if (spacey_correction) {
    for (int i = 0; i < dim; ++i) {
      for (int j = 0; j < dim; ++j) {
	double max = 0.0;
	for (int k = 0; k < dim; ++k) {
	  max = std::max(X(i, j, k), max);
	  if (max > 0.0) { break; }
	}
	if (max == 0.0) {
	  for (int k = 0; k < dim; ++k) {
	    X.Set(i, j, k, 1);
	  }
	}
      }
    }
  }
    
  NormalizeStochastic(X);
  return X;
}

Tensor3 GradientDescent(std::vector< std::vector<int> >& seqs) {
#if 1
  Tensor3 X = EmpiricalSecondOrder(seqs, true);
#else
  Tensor3 X(MaximumIndex(seqs) + 1);
  X.SetGlobalValue(1.0);
  NormalizeStochastic(X);
#endif
  double curr_ll = LogLikelihoodTrain(seqs, X);
  double generalization_error = SpaceyRMSE(seqs, X);
  std::cerr << "Starting log likelihood: " << curr_ll << std::endl;
  std::cerr << "Generalization RMSE: " << generalization_error << std::endl;

  int niter = 5000;
  double starting_step_size = 1;
  for (int iter = 0; iter < niter; ++iter) {
    double step_size = starting_step_size; // / (iter + 1);
    Tensor3 Grad = Gradient(seqs, X);
    Tensor3 Y = UpdateAndProject(X, Grad, step_size);
    double update_ll = LogLikelihoodTrain(seqs, Y);
    if (update_ll > curr_ll) {
      X = Y;
      curr_ll = update_ll;
      double gen_rmse = SpaceyRMSE(seqs, X);
      std::cerr << curr_ll << " " << step_size
		<< " (" << gen_rmse << ")" << std::endl;
    } else {
      starting_step_size *= 0.5;
    }
    if (starting_step_size < 1e-15) { break; }
  }
  return X;
}

double SecondOrderRMSE(std::vector< std::vector<int> >& seqs) {
  Tensor3 P = EmpiricalSecondOrder(seqs, false);
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

double SecondOrderUniformCollapseRMSE(std::vector< std::vector<int> >& seqs) {
  Tensor3 P = EmpiricalSecondOrder(seqs, false);
  // Collapse
  Matrix2 Pc(P.dim());
  Pc.SetGlobalValue(0.0);
  for (int i = 0; i < P.dim(); ++i) {
    for (int j = 0; j < P.dim(); ++j) {
      for (int k = 0; k < P.dim(); ++k) {
	Pc(i, j) = Pc.Get(i, j) + P(i, j, k) / P.dim();
      }
    }
  }

  // Compute RMSE
  double err = 0.0;
  int num = 0;
  for (auto& seq : seqs) {
    for (int l = EndingIndex(seq); l < seq.size(); ++l) {
      int j = seq[l - 1];
      int i = seq[l];
      double val = 1 - Pc(i, j);
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
	    << SecondOrderRMSE(seqs) << std::endl
	    << "Second order uniform collapse RMSE: "
	    << SecondOrderUniformCollapseRMSE(seqs) << std::endl;
  Tensor3 PSRS = GradientDescent(seqs);
  Tensor3 PSO = EmpiricalSecondOrder(seqs, false);
  int N = PSRS.dim();
  double diff = L1Diff(PSRS, PSO) / (N * N);
  std::cout << "Difference: " << diff << std::endl;
}
