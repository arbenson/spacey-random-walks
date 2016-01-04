#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "common_srw.hpp"
#include "tensor3.hpp"

static bool verbose_flag = true;

Tensor3 EmpiricalSecondOrder(const std::vector< std::vector<int> >& seqs) {
  int dim = MaximumIndex(seqs) + 1;
  Tensor3 X(dim);
  X.SetGlobalValue(0);
  for (auto& seq : seqs) {
    for (int l = 1; l < seq.size(); ++l) {
      int k = 0;  // Starts at zero by default
      if (l > 1) {
        k = seq[l - 2];
      }
      int j = seq[l - 1];
      int i = seq[l];
      X(i, j, k) = X.Get(i, j, k) + 1;
    }
  }
  NormalizeStochastic(X);
  return X;
}

Matrix2 EmpiricalFirstOrder(const std::vector< std::vector<int> >& seqs) {
  // Fill in empirical transitions
  int dim = MaximumIndex(seqs) + 1;
  Matrix2 X(dim);
  X.SetGlobalValue(0);
  for (auto& seq : seqs) {
    for (int l = 0; l < seq.size(); ++l) {
      int j = 0;  // Starts at zero by default
      if (l > 0) {
	j = seq[l - 1];
      }
      int i = seq[l];
      X(i, j) = X.Get(i, j) + 1;
    }
  }

  // Normalize to stochastic
  for (int j = 0; j < dim; ++j) {
    std::vector<double> col = X.GetColumn(j);
    std::vector<double> ncol = Normalized(col);
    X.SetColumn(j, ncol);
  }
  return X;
}

Tensor3 Gradient(const std::vector< std::vector<int> >& seqs,
                 const Tensor3& P) {
  Tensor3 G(P.dim());
  G.SetGlobalValue(0.0);
  for (auto& seq : seqs) {
    std::vector<int> history(P.dim(), 1);
    history[seq[0]] += 1;
    for (int l = 1; l < seq.size(); ++l) {
      std::vector<double> occupancy = Normalized(history);
      int i = seq[l];
      int j = seq[l - 1];

      double sum = 0.0;
      for (int k = 0; k < P.dim(); ++k) {
        sum += occupancy[k] * P(i, j, k);
      }
      for (int k = 0; k < P.dim(); ++k) {
        G(i, j, k) = G.Get(i, j, k) + occupancy[k] / sum;
      }
      history[i] += 1;
    }
  }
  return G;
}

double LogLikelihood(const Tensor3& P, const std::vector< std::vector<int> >& seqs) {
  double ll = 0.0;
  for (auto& seq : seqs) {
    std::vector<int> history(P.dim(), 1);
    history[seq[0]] += 1;
    for (int l = 1; l < seq.size(); ++l) {
      std::vector<double> occupancy = Normalized(history);
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

Tensor3 SRSGradientUpdate(const Tensor3& X, double step_size,
			  const Tensor3& gradient) {
  int dim = X.dim();
  Tensor3 Y(dim);
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      for (int k = 0; k < dim; ++k) {
        Y(i, j, k) = X(i, j, k) + step_size * gradient(i, j, k);
      }
    }
  }
  ProjectColumnsOntoSimplex(Y);
  return Y;
}

Tensor3 EstimateSRS(const std::vector< std::vector<int> >& seqs) {
  int dim = MaximumIndex(seqs) + 1;
  Tensor3 X = EmpiricalSecondOrder(seqs);
  double curr_ll = LogLikelihood(X, seqs);

  int niter = 40000;
  double starting_step_size = 1;
  for (int iter = 0; iter < niter; ++iter) {
    double step_size = starting_step_size / (iter + 1);
    Tensor3 grad = Gradient(seqs, X);
    Tensor3 Y = SRSGradientUpdate(X, step_size, grad);
    double next_ll = LogLikelihood(Y, seqs);
    if (next_ll > curr_ll) {
      X = Y;
      curr_ll = next_ll;
      if (iter % 1000 == 0 && verbose_flag) {
	std::cerr << curr_ll << " " << iter << " " << step_size << std::endl;
      }
    } else {
      starting_step_size *= 0.5;
    }
  }
  return X;
}

double SecondOrderLogLikelihood(const Tensor3& P,
                                const std::vector< std::vector<int> >& seqs) {
  double ll = 0.0;
  for (auto& seq : seqs) {
    for (int l = 1; l < seq.size(); ++l) {
      // Always starts at 0
      int k = 0;
      if (l > 1) { k = seq[l - 2]; }
      int j = seq[l - 1];
      int i = seq[l];
      ll += log(P(i, j, k));
    }
  }
  return ll;
}

Tensor3 ReadTensor(std::string filename) {
  std::vector<int> i_ind, j_ind, k_ind;
  std::vector<double> vals;

  std::ifstream infile(filename);
  std::string line;
  while (std::getline(infile, line)) {
    int i, j, k;
    double val = 0.0;
    char delim;

    std::istringstream iss(line);
    iss >> i;
    iss >> j;
    iss >> k;
    iss >> val;

    i_ind.push_back(i);
    j_ind.push_back(j);
    k_ind.push_back(k);
    vals.push_back(val);
  }
  
  // Get the maximum index
  int max_ind = 0;
  for (int l = 0; l < i_ind.size(); ++l) {
    max_ind = std::max(max_ind, i_ind[l]);
    max_ind = std::max(max_ind, j_ind[l]);
    max_ind = std::max(max_ind, k_ind[l]);
  }

  // Put the values in P (treat as sparse indices).
  Tensor3 P(max_ind + 1);
  P.SetGlobalValue(0.0);
  for (int l = 0; l < i_ind.size(); ++l) {
    P(i_ind[l], j_ind[l], k_ind[l]) = vals[l];
  }

  return P;
}

double SpaceyRMSE(const std::vector< std::vector<int> >& seqs,
                  const Tensor3& P) {
  double err = 0.0;
  int num = 0;
  for (auto& seq : seqs) {
    std::vector<int> history(P.dim(), 1);
    history[seq[0]] += 1;
    for (int l = 1; l < seq.size(); ++l) {
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

double SecondOrderRMSE(const std::vector< std::vector<int> >& seqs,
		       const Tensor3& P) {
  double err = 0.0;
  int num = 0;
  for (auto& seq : seqs) {
    for (int l = 1; l < seq.size(); ++l) {
      int i = seq[l];
      int j = seq[l - 1];
      int k = 0; // Starts at 0
      if (l >= 2) { k = seq[l - 2]; }

      double val = 1 - P(i, j, k);
      err += val * val;
      ++num;
    }
  }
  return sqrt(err / num);
}

double FirstOrderRMSE(const std::vector< std::vector<int> >& seqs,
		      const Matrix2& P) {
  double err = 0.0;
  int num = 0;
  for (auto& seq : seqs) {
    for (int l = 1; l < seq.size(); ++l) {
      int i = seq[l];
      int j = seq[l - 1];
      double val = 1 - P(i, j);
      err += val * val;
      ++num;
    }
  }
  return sqrt(err / num);
}

int main(int argc, char **argv) {
  std::vector< std::vector<int> > train_seqs, test_seqs;
  ReadSequences(argv[1], train_seqs);
  ReadSequences(argv[2], test_seqs);
  Tensor3 P = ReadTensor(argv[3]);

#if 0
  std::vector< std::vector<int> > train_seqs2;
  train_seqs2.push_back(train_seqs[0]);
#endif
  Tensor3 PSO = EmpiricalSecondOrder(train_seqs);
  Matrix2 PFO = EmpiricalFirstOrder(train_seqs);

  double err1 = SpaceyRMSE(test_seqs, P);
  double err2 = SecondOrderRMSE(test_seqs, PSO);
  double err3 = FirstOrderRMSE(test_seqs, PFO);

  std::cout << "Spacey:       " << err1 << std::endl
	    << "Second-order: " << err2 << std::endl
	    << "First-order:  " << err3 << std::endl;
}
