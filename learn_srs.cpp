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

int StartingIndex(const std::vector<int>& seq) {
  return floor((fraction_init) * seq.size());
}

int EndingIndex(const std::vector<int>& seq) {
  return floor((fraction_init + fraction_train) * seq.size());
}

std::vector<int> InitializeHistory(const std::vector<int>& seq, int dimension) {
  std::vector<int> history(dimension, 1);
  for (int i = 0; i < StartingIndex(seq); ++i) {
    history[seq[i]] += 1;
  }
  return history;
}

double SpaceyPrecisionAtK(const std::vector< std::vector<int> >& seqs,
			  const Tensor3& P, int K) {
  int correct = 0;
  int num = 0;
  for (auto& seq : seqs) {
    std::vector<int> history(P.dim(), 1);
    for (int i = 0; i < EndingIndex(seq); ++i) {
      history[seq[i]] += 1;
    }
    for (int l = EndingIndex(seq); l < seq.size(); ++l) {
      auto occupancy = Normalized(history);
      int i = seq[l];
      int j = seq[l - 1];
      std::vector<double> probs(P.dim(), 0.0);
      for (int r = 0; r < occupancy.size(); ++r) {
	for (int ii = 0; ii < P.dim(); ++ii) {
	  probs[ii] += occupancy[r] * P(ii, j, r);
	}
      }
      if (InTopK(probs, i, K)) {
	++correct;
      }
      ++num;
      history[i] += 1;
    }
  }
  return static_cast<double>(correct) / num;
}

Tensor3 Gradient(const std::vector< std::vector<int> >& seqs,
                 const Tensor3& P) {
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

double LogLikelihoodTrain(const std::vector< std::vector<int> >& seqs,
                          const Tensor3& P) {
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

double SpaceyRMSE(const std::vector< std::vector<int> >& seqs,
                  const Tensor3& P) {
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

Tensor3 UpdateAndProject(const Tensor3& X, const Tensor3& G, double step_size) {
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
  ProjectColumnsOntoSimplex(Y);
  return Y;
}

Tensor3 EmpiricalSecondOrder(const std::vector< std::vector<int> >& seqs,
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

Tensor3 EstimateSRS(const std::vector< std::vector<int> >& seqs) {
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
  double srs_pak1 = SpaceyPrecisionAtK(seqs, X, 1);
  double srs_pak2 = SpaceyPrecisionAtK(seqs, X, 2);
  double srs_pak3 = SpaceyPrecisionAtK(seqs, X, 3);
  std::cerr << "Precisions@{1,2,3}: " 
	    << srs_pak1 << " "
	    << srs_pak2 << " "
	    << srs_pak3 << std::endl;

  int niter = 5000;
  double starting_step_size = 1e-6;
  for (int iter = 0; iter < niter; ++iter) {
    double step_size = starting_step_size;
    Tensor3 Grad = Gradient(seqs, X);
    Tensor3 Y = UpdateAndProject(X, Grad, step_size);
    double update_ll = LogLikelihoodTrain(seqs, Y);

    if (update_ll > curr_ll) {
      X = Y;
      curr_ll = update_ll;
      double gen_rmse = SpaceyRMSE(seqs, Y);
      std::cerr << update_ll << " " << step_size
		<< " (" << gen_rmse << ")" << std::endl;
      srs_pak1 = SpaceyPrecisionAtK(seqs, Y, 1);
      srs_pak2 = SpaceyPrecisionAtK(seqs, Y, 2);
      srs_pak3 = SpaceyPrecisionAtK(seqs, Y, 3);
      std::cerr << "Precisions@{1,2,3}: "
		<< srs_pak1 << " "
		<< srs_pak2 << " "
		<< srs_pak3 << std::endl;
    } else {
      starting_step_size *= 0.5;
    }
    if (starting_step_size < 1e-15) { break; }
  }
  return X;
}

double SecondOrderRMSE(Tensor3& P, std::vector< std::vector<int> >& seqs) {
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

double SecondOrderPrecisionAtK(Tensor3& P, std::vector< std::vector<int> >& seqs, int K) {
  int correct = 0;
  int num = 0;
  for (auto& seq : seqs) {
    for (int l = EndingIndex(seq); l < seq.size(); ++l) {
      int r = seq[l - 2];
      int j = seq[l - 1];
      auto vec = P.GetSlice1(j, r);
      int i = seq[l];
      if (InTopK(vec, i, K)) {
	++correct;
      }
      ++num;
    }
  }
  return static_cast<double>(correct) / num;
}

double SecondOrderUniformCollapseRMSE(Tensor3& P, std::vector< std::vector<int> >& seqs) {
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

Matrix2 EmpiricalFirstOrder(std::vector< std::vector<int> >& seqs) {
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
  return X;
}

double FirstOrderRMSE(Matrix2& P, std::vector< std::vector<int> >& seqs) {
  // Compute Log Likelihood
  double err = 0;
  int num = 0;
  for (auto& seq : seqs) {
    for (int l = EndingIndex(seq); l < seq.size(); ++l) {
      int j = seq[l - 1];
      int i = seq[l];
      double val = (1 - P(i, j));
      err += val * val;
      ++num;
    }
  }
  return sqrt(err / num);
}

double FirstOrderPrecisionAtK(Matrix2& P, std::vector< std::vector<int> >& seqs, int K) {
  // Compute Log Likelihood
  int correct = 0;
  int num = 0;
  for (auto& seq : seqs) {
    for (int l = EndingIndex(seq); l < seq.size(); ++l) {
      int j = seq[l - 1];
      std::vector<double> col = P.GetColumn(j);
      int i = seq[l];
      if (InTopK(col, i, K)) {
	++correct;
      }
      ++num;
    }
  }
  return static_cast<double>(correct) / num;
}

int main(int argc, char **argv) {
  std::vector< std::vector<int> > seqs;
  ReadSequences(argv[1], seqs);

  Tensor3 PSO = EmpiricalSecondOrder(seqs, false);
  Matrix2 PFO = EmpiricalFirstOrder(seqs);

  std::cerr << seqs.size() << " sequences." << std::endl;
  std::cerr << "First order RMSE: "
            << FirstOrderRMSE(PFO, seqs) << std::endl
            << "Second order RMSE: "
            << SecondOrderRMSE(PSO, seqs) << std::endl
            << "Second order uniform collapse RMSE: "
            << SecondOrderUniformCollapseRMSE(PSO, seqs) << std::endl;

  double so_pak1 = SecondOrderPrecisionAtK(PSO, seqs, 1);
  double so_pak2 = SecondOrderPrecisionAtK(PSO, seqs, 2);
  double so_pak3 = SecondOrderPrecisionAtK(PSO, seqs, 3);
  std::cerr << "SO precisions@{1,2,3}: "
	    << so_pak1 << " "
	    << so_pak2 << " "
	    << so_pak3 << std::endl;

  double fo_pak1 = FirstOrderPrecisionAtK(PFO, seqs, 1);
  double fo_pak2 = FirstOrderPrecisionAtK(PFO, seqs, 2);
  double fo_pak3 = FirstOrderPrecisionAtK(PFO, seqs, 3);
  std::cerr << "FO precisions@{1,2,3}: "
	    << fo_pak1 << " "
	    << fo_pak2 << " "
	    << fo_pak3 << std::endl;

  Tensor3 PSRS = EstimateSRS(seqs);
  int N = PSRS.dim();
  double diff = L1Diff(PSRS, PSO) / (N * N);

  std::cout << "Difference: " << diff << std::endl;

  int dimension = PSRS.dim();
  assert(dimension == PSO.dim());
  //for (int j = 0; j < dimension; ++j) {
  int j = 2;
  for (int k = 0; k < dimension; ++k) {
    if (k == 8 || k == 21 || k == 28 || k == 33) {
      for (int i = 0; i < dimension; ++i) {
	std::cout << PSRS(i, j, k) << " "
		  << PSO(i, j, k) << " "
		  << i << " "
		  << j << " "
		  << k << std::endl;
      }
    }
  }
}
