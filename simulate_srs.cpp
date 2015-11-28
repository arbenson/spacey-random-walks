#include <iostream>
#include <random>

#include <getopt.h>

#include "common_srs.hpp"
#include "tensor3.hpp"

static int verbose_flag = 0;
static int problem_dimension = 4;
static int number_of_simulated_sequences = 25;
static int size_of_simulated_sequence = 1000;


// || vec(P1) - vec(P2) ||_1
double L1Diff(Tensor3& P1, Tensor3& P2) {
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

// || v1 - v2 ||_1
double L1Diff(std::vector<double>& v1, std::vector<double>& v2) {
  double diff = 0.0;
  assert(v1.size() == v2.size());
  for (int i = 0; i < v1.size(); ++i) {
    diff += std::abs(v1[i] - v2[i]);
  }
  return diff;
}

std::vector<double> Apply(Tensor3& P, std::vector<double>& x) {
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

// y = P x^2
std::vector<double> TensorApply(Tensor3& P, std::vector<double>& x) {
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

std::vector<double> Stationary(Tensor3& P) {
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

std::vector<double> StationaryMarginals(Tensor3& P) {
  std::vector<double> st = Stationary(P);
  int dim = P.dim();
  std::vector<double> marginals(dim, 0.0);
  for (int i = 0; i < st.size(); ++i) {
    int marginal_ind = (i % dim);
    marginals[marginal_ind] += st[i];
  }
  return marginals;
}

std::vector<double> SpaceyStationary(Tensor3& P) {
  int dim = P.dim();
  std::vector<double> x(dim, 1.0 / dim);
  int max_iter = 1000;
  double tol = 1e-12;
  for (int iter = 0; iter < max_iter; ++iter) {
    std::vector<double> x_next = TensorApply(P, x);
    // Check the difference
    double diff = L1Diff(x_next, x);
    for (int j = 0; j < x.size(); ++j) {
      x[j] = 0.99 * x_next[j] + 0.01 * x[j];
    }
    x = Normalized(x);
    // Stop if difference is small enough
    if (diff < tol) { break; }
  }
  return x;
}

Tensor3 EmpiricalSecondOrder(std::vector< std::vector<int> >& seqs) {
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

Tensor3 Gradient(std::vector< std::vector<int> >& seqs,
                 Tensor3& P) {
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

double LogLikelihood(Tensor3& P, std::vector< std::vector<int> >& seqs) {
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

// Form uniform random transition probability tensor.
Tensor3 RandomTPT(int dimension) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0, 1);
  Tensor3 P(dimension);
  for (int i = 0; i < dimension; ++i) {
    for (int j = 0; j < dimension; ++j) {
      for (int k = 0; k < dimension; ++k) {
        // Random uniform value
        P(i, j, k) = dis(gen);
      }
    }
  }
  NormalizeStochastic(P);
  return P;
}

Tensor3 SRSGradientUpdate(Tensor3& X, double step_size,
                          Tensor3& gradient) {
  int dim = X.dim();
  Tensor3 Y(dim);
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      for (int k = 0; k < dim; ++k) {
        Y(i, j, k) = X(i, j, k) + step_size * gradient(i, j, k);
      }
    }
  }
  Project(Y);
  return Y;
}

Tensor3 EstimateSRS(std::vector< std::vector<int> >& seqs) {
  int dim = MaximumIndex(seqs) + 1;
#if 0
  Tensor3 X(dim);
  X.SetGlobalValue(1.0);
  NormalizeStochastic(X);
#else
  Tensor3 X = EmpiricalSecondOrder(seqs);
#endif
  double curr_ll = LogLikelihood(X, seqs);

  int niter = 10000;
  double starting_step_size = 1e-5;
  for (int iter = 0; iter < niter; ++iter) {
    double step_size = starting_step_size / (iter + 1);
    Tensor3 grad = Gradient(seqs, X);
    Tensor3 Y = SRSGradientUpdate(X, step_size, grad);
    double next_ll = LogLikelihood(Y, seqs);
    if (next_ll > curr_ll) {
      X = Y;
      curr_ll = next_ll;
      if (iter % 100 == 0 && verbose_flag) {
	std::cerr << curr_ll << " " << step_size << std::endl;
      }
    }
  }
  return X;
}

void Simulate(Tensor3& P, std::vector< std::vector<int> >& seqs,
              int num_seqs, int num_samples) {
  int dim = P.dim();
  seqs.clear();
  for (int seq_ind = 0; seq_ind < num_seqs; ++seq_ind) {
    std::vector<int> history(dim, 1);
    int j = 0;
    std::vector<int> seq(num_samples);
    for (int sample_ind = 0; sample_ind < num_samples; ++sample_ind) {
      // Choose from history
      std::vector<double> occupancy = Normalized(history);
      int k = Choice(occupancy);

      // Follow transition
      int i = Choice(P.GetSlice1(j, k));

      // Update history
      history[i] += 1;
      j = i;
      seq[sample_ind] = i;
    }
    seqs.push_back(seq);
  }
}

double SecondOrderLogLikelihood(Tensor3& P,
                                std::vector< std::vector<int> >& seqs) {
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

void HandleOptions(int argc, char **argv) {
  static struct option long_options[] =
    {
      {"verbose",       no_argument, &verbose_flag, 1},
      {"dimension",     required_argument, 0, 'd'},
      {"numsequences",  required_argument, 0, 'n'},
      {"sequence",      required_argument, 0, 's'},
      {0, 0, 0, 0}
    };

  int c;

  while (1) {
      int option_index = 0;
      c = getopt_long (argc, argv, "d:n:s:",
                       long_options, &option_index);
      /* Detect the end of the options. */
      if (c == -1) {
        break;
      }

      switch (c) {
        case 0:
          /* If this option set a flag, do nothing else now. */
          if (long_options[option_index].flag != 0)
            break;
        case 'd':
	  problem_dimension = atoi(optarg);
          break;
        case 'n':
	  number_of_simulated_sequences = atoi(optarg);
          break;
        case 's':
	  size_of_simulated_sequence = atoi(optarg);
          break;
        default:
          abort();
      }
  }

}


int main(int argc, char **argv) {
  HandleOptions(argc, argv);
  int N = problem_dimension;
  int num_seqs = number_of_simulated_sequences;
  int samples_per_seq = size_of_simulated_sequence;

  std::cout << N << " "
	    << num_seqs << " "
	    << samples_per_seq << std::endl;


  std::vector< std::vector<int> > seqs;
  Tensor3 P = RandomTPT(N);
  Simulate(P, seqs, num_seqs, samples_per_seq);

  double oracle_ll = LogLikelihood(P, seqs);
  Tensor3 PSO = EmpiricalSecondOrder(seqs);
  double empirical_ll = LogLikelihood(PSO, seqs);
  double so_ll = SecondOrderLogLikelihood(PSO, seqs);
  Tensor3 PSRS = EstimateSRS(seqs);
  double srs_ll = LogLikelihood(PSRS, seqs);

  std::cout << "Oracle LL:       " << oracle_ll    << std::endl
	    << "Empirical LL:    " << empirical_ll << std::endl
	    << "SRS LL:          " << srs_ll       << std::endl
	    << "Second-order LL: " << so_ll        << std::endl
	    << std::endl;

  int num_total = (samples_per_seq - 1) * num_seqs;
  double ll_diff1 = exp((empirical_ll - oracle_ll) / num_total);
  double ll_diff2 = exp((srs_ll - oracle_ll) / num_total);
  std::cout << "LL(empirical to oracle) = " << ll_diff1 << std::endl
	    << "LL(learned to oracle)   = " << ll_diff2 << std::endl
	    << std::endl;
    
  double P_diff1 = L1Diff(P, PSO) / (N * N);
  double P_diff2 = L1Diff(P, PSRS) / (N * N);
  std::cout << "|| vec(P) - vec(PSO) ||_1  = " << P_diff1 << std::endl
	    << "|| vec(P) - vec(PSRS) ||_1 = " << P_diff2 << std::endl
	    << std::endl;

  std::vector<double> srs_st_orig       = SpaceyStationary(P);
  std::vector<double> srs_st_recovered  = SpaceyStationary(PSRS);
  std::vector<double> srs_st_so         = SpaceyStationary(PSO);
  std::vector<double> marginal_st_so    = StationaryMarginals(PSO);
  double st_diff1 = L1Diff(srs_st_orig, marginal_st_so);
  double st_diff2 = L1Diff(srs_st_orig, srs_st_recovered);
  double st_diff3 = L1Diff(srs_st_orig, srs_st_so);
  std::cout << "|| SRS(P) - MarkovMarginal(PSO) ||_1 = " << st_diff1 << std::endl
	    << "|| SRS(P) - SRS(PSRS) ||_1           = " << st_diff2 << std::endl
	    << "|| SRS(P) - SRS(PSO) ||_1            = " << st_diff3 << std::endl;
}
