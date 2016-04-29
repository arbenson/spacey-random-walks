#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <getopt.h>

#include "common_srw.hpp"
#include "hypermatrix.hpp"

static int max_iter = 1000;
static double starting_step_size = 1;
static double step_size_reduction = 0.5;
static double minimum_step_size = 1e-16;
static int update_frequency = 1000;
static std::string train_file = "";
static std::string test_file = "";
static std::string hypermatrix_input_file = "";
static std::string hypermatrix_output_file = "P.txt";

// Get the empirical (MLE) second-order Markov chain transition probabilities.
// returns X such that X(i, j, k) = Pr((k, j) --> (j, i))
DblCubeHypermatrix EmpiricalSecondOrder(const std::vector< std::vector<int> >& seqs) {
  int dim = MaximumIndex(seqs) + 1;
  DblCubeHypermatrix X(dim);
  X.SetGlobalValue(0);
  for (auto& seq : seqs) {
    for (int l = 2; l < seq.size(); ++l) {
      int k = seq[l - 2];
      int j = seq[l - 1];
      int i = seq[l];
      X(i, j, k) = X.Get(i, j, k) + 1;
    }
  }
  NormalizeStochastic(X);
  return X;
}

// Get the empirical (MLE) first-order Markov chain transition probabilities.
DblSquareMatrix EmpiricalFirstOrder(const std::vector< std::vector<int> >& seqs) {
  // Fill in empirical transitions
  int dim = MaximumIndex(seqs) + 1;
  DblSquareMatrix X(dim);
  X.SetGlobalValue(0);
  for (auto& seq : seqs) {
    for (int l = 1; l < seq.size(); ++l) {
      int j = seq[l - 1];
      int i = seq[l];
      X(i, j) = X.Get(i, j) + 1;
    }
  }

  // Normalize to stochastic
  for (int j = 0; j < dim; ++j) {
    std::vector<double> col = X.GetColumn(j);

    // If the column sum is 0, guess randomly
    std::vector<double> ncol(dim, 1.0 / dim);
    if (Sum(col) > 0) {
      ncol = Normalized(col);
    }
    X.SetColumn(j, ncol);
  }
  return X;
}

// Get the empirical (MLE) zero-order Markov chain transition probabilities,
// which are just the marginal probabilities of being at each state.
std::vector<double> EmpiricalZerothOrder(const std::vector< std::vector<int> >& seqs) {
  // Fill in empirical transitions
  int dim = MaximumIndex(seqs) + 1;
  std::vector<int> counts(dim, 0);
  int total = 0;
  for (auto& seq : seqs) {
    for (int place : seq) {
      ++counts[place];
      ++total;
    }
  }

  // Normalize to stochastic
  std::vector<double> x(dim);
  for (int j = 0; j < dim; ++j) {
    x[j] = static_cast<double>(counts[j]) / total;
  }
  return x;
}

// Get the gradient of the likelihood for the spacey random walk model
DblCubeHypermatrix Gradient(const std::vector< std::vector<int> >& seqs,
			    const DblCubeHypermatrix& P) {
  DblCubeHypermatrix G(P.dim());
  G.SetGlobalValue(0.0);
  for (auto& seq : seqs) {
    std::vector<int> history(P.dim(), 1);
    // Count the first one
    if (seq.size() <= 1) { continue; }
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

// Compute the spacey random walk log likelihood for the spacey random
// walk model with transition probabilities P.
double LogLikelihood(const std::vector< std::vector<int> >& seqs,
		     const DblCubeHypermatrix& P) {
  double ll = 0.0;
  for (auto& seq : seqs) {
    std::vector<int> history(P.dim(), 1);
    if (seq.size() <= 1) {
      continue;
    }
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

// Given the current transition probabilities X, the gradient, and the step size,
// compute the next iterative of projected gradient descent.
DblCubeHypermatrix SRWGradientUpdate(const DblCubeHypermatrix& X,
				     const DblCubeHypermatrix& gradient, double step_size) {
  int dim = X.dim();
  DblCubeHypermatrix Y(dim);
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

// Estimate the spacey random walk transition probabilities using projected
// gradient descent.
DblCubeHypermatrix EstimateSRW(const std::vector< std::vector<int> >& seqs) {
  DblCubeHypermatrix X = EmpiricalSecondOrder(seqs);
  double curr_ll = LogLikelihood(seqs, X);

  double step_size = starting_step_size;
  for (int iter = 0; iter < max_iter; ++iter) {
    DblCubeHypermatrix grad = Gradient(seqs, X);
    DblCubeHypermatrix Y = SRWGradientUpdate(X, grad, step_size);
    double next_ll = LogLikelihood(seqs, Y);
    if (next_ll > curr_ll) {
      X = Y;
      curr_ll = next_ll;
      if (iter % update_frequency == 0) {
        std::cerr << curr_ll << " " << iter << " " << step_size << std::endl;
      }
    } else {
      // decrease step size
      step_size *= step_size_reduction;
    }
    // Stop if the step size becomes too small.
    if (step_size < minimum_step_size) {
      break;
    }
  }
  return X;
}

// Read a third-order sparse hypermatrix from a text file.  Each line of the
// file looks like
//
//       i j k v
// 
// which says that the (i, j, k) entry has value v.  All triplets not in the
// file are assumed to be 0.
DblCubeHypermatrix ReadHypermatrix(std::string filename) {
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
  DblCubeHypermatrix P(max_ind + 1);
  P.SetGlobalValue(0.0);
  for (int l = 0; l < i_ind.size(); ++l) {
    P(i_ind[l], j_ind[l], k_ind[l]) = vals[l];
  }

  return P;
}

// RMSE for the spacey random walk model
double SpaceyRMSE(const std::vector< std::vector<int> >& seqs,
                  const DblCubeHypermatrix& P) {
  double err = 0.0;
  int num = 0;
  for (auto& seq : seqs) {
    std::vector<int> history(P.dim(), 1);
    if (seq.size() <= 2) { continue; }
    history[seq[0]] += 1;
    history[seq[1]] += 1;
    for (int l = 2; l < seq.size(); ++l) {
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

// RMSE for the second-order Markov chain model
double SecondOrderRMSE(const std::vector< std::vector<int> >& seqs,
                       const DblCubeHypermatrix& P) {
  double err = 0.0;
  int num = 0;
  for (auto& seq : seqs) {
    for (int l = 2; l < seq.size(); ++l) {
      int i = seq[l];
      int j = seq[l - 1];
      int k = seq[l - 2];  // Starts at 0 by default

      double val = 1 - P(i, j, k);
      err += val * val;
      ++num;
    }
  }
  return sqrt(err / num);
}

// RMSE for the first-order Markov chain model
double FirstOrderRMSE(const std::vector< std::vector<int> >& seqs,
                      const DblSquareMatrix& P) {
  double err = 0.0;
  int num = 0;
  for (auto& seq : seqs) {
    for (int l = 2; l < seq.size(); ++l) {
      int i = seq[l];
      int j = seq[l - 1];
      double val = 1 - P(i, j);
      err += val * val;
      ++num;
    }
  }
  return sqrt(err / num);
}

// RMSE for the zeroth-order Markov chain model (i.e., marginals)
double ZerothOrderRMSE(const std::vector< std::vector<int> >& seqs,
		       const std::vector<double>& p) {
  double err = 0.0;
  int num = 0;
  for (auto& seq : seqs) {
    for (int l = 2; l < seq.size(); ++l) {
      double val = p[seq[l]];
      err += 1 - val * val;
      ++num;
    }
  }
  return sqrt(err / num);
}

void HandleOptions(int argc, char **argv) {
  static struct option long_options[] =
    {
      {"max_iter",            required_argument, 0, 'n'},
      {"starting_step_size",  required_argument, 0, 's'},
      {"step_size_reduction", required_argument, 0, 'r'},
      {"minimum_step_size",   required_argument, 0, 'm'},
      {"train_file",          required_argument, 0, 't'},
      {"test_file",           required_argument, 0, 'e'},
      {"hypermatrix_input_file",   required_argument, 0, 'p'},
      {"hypermatrix_output_file",  required_argument, 0, 'o'},
      {"update_frequency",    required_argument, 0, 'u'},
      {0, 0, 0, 0}
    };

  int c;
  while (1) {
    int option_index = 0;
    c = getopt_long (argc, argv, "n:s:r:m:t:p:e:o:u:",
                     long_options, &option_index);
    // Detect the end of the options.
    if (c == -1) {
      break;
    }

    switch (c) {
    case 0:
      // If this option set a flag, do nothing else now.
      if (long_options[option_index].flag != 0) {
        break;
      }
    case 'n':
      max_iter = atoi(optarg);
      break;
    case 's':
      starting_step_size = atof(optarg);
      break;
    case 'r':
      step_size_reduction = atof(optarg);
      break;
    case 'm':
      minimum_step_size = atof(optarg);
    case 't':
      train_file = std::string(optarg);
      break;
    case 'e':
      test_file = std::string(optarg);
      break;
    case 'p':
      hypermatrix_input_file = std::string(optarg);
      break;
    case 'o':
      hypermatrix_output_file = std::string(optarg);
      break;
    case 'u':
      update_frequency = atoi(optarg);
      break;
    default:
      abort();
    }
  }
}

int main(int argc, char **argv) {
  HandleOptions(argc, argv);
  if (train_file.size() == 0) {
    std::cerr << "Must provide training data (-t)." << std::endl;
    abort();
  }
  if (test_file.size() == 0) {
    std::cerr << "Must provide test data (-e)." << std::endl;
    abort();
  }
  std::vector< std::vector<int> > train_seqs, test_seqs;
  ReadSequences(train_file, train_seqs);
  ReadSequences(test_file, test_seqs);
  DblCubeHypermatrix P;
  if (hypermatrix_input_file.size() > 0) {
    P = ReadHypermatrix(hypermatrix_input_file);
  }

  std::vector<double> marginal = EmpiricalZerothOrder(train_seqs);
  DblSquareMatrix PFO  = EmpiricalFirstOrder(train_seqs);
  DblCubeHypermatrix PSO  = EmpiricalSecondOrder(train_seqs);
  DblCubeHypermatrix PSRW = EstimateSRW(train_seqs);


  double err1 = 0;
  if (hypermatrix_input_file.size() > 0) {
    err1 = SpaceyRMSE(test_seqs, P);
  }
  double err2 = SpaceyRMSE(test_seqs, PSRW);
  double err3 = SpaceyRMSE(test_seqs, PSO);
  double err4 = SecondOrderRMSE(test_seqs, PSO);
  double err5 = FirstOrderRMSE(test_seqs, PFO);
  double err6 = ZerothOrderRMSE(test_seqs, marginal);

  // RMSEs on the test data
  if (hypermatrix_input_file.size() > 0) {
    std::cout << "Spacey (true):      " << err1 << std::endl;
  }
  std::cout << "Spacey (estimated): " << err2 << std::endl;
  std::cout << "Spacey (empirical): " << err3 << std::endl;
  std::cout << "Second-order:       " << err4 << std::endl;
  std::cout << "First-order:        " << err5 << std::endl;
  std::cout << "Zeroth-order:       " << err6 << std::endl;

  // Error in parameter recovery
  if (hypermatrix_input_file.size() > 0) {
    int N = P.dim();
    double diff1 = L1Diff(PSRW, P) / (N * N);
    double diff2 = L1Diff(PSO,  P) / (N * N);
    std::cout << "|| PSRW - P ||_1 / N^2: " << diff1 << std::endl;
    std::cout << "|| PSO - P ||_1 / N^2 : " << diff2 << std::endl;
  }

  WriteHypermatrix(PSRW, hypermatrix_output_file);  
}
