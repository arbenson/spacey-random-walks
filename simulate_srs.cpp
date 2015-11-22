#include <iostream>
#include <random>

#include "common_srs.hpp"
#include "tensor3.hpp"

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
    for (int l = 0; l < seq.size(); ++l) {
      std::vector<double> occupancy = Normalized(history);
      int i = seq[l];
      int j = 0;  // always start at 0
      if (l > 0) {
	int j = seq[l - 1];
      }
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
    for (int l = 0; l < seq.size(); ++l) {
      std::vector<double> occupancy = Normalized(history);
      int i = seq[l];
      int j = 0;  // always start at 0
      if (l > 0) {
	int j = seq[l - 1];
      }
      double sum = 0.0;
      for (int k = 0; k < occupancy.size(); ++k) {
	sum += occupancy[k] * P(i, j, k);
      }
      ll += log(sum);
    }
  }
  return ll;
}


// Sample from a discrete probability distribution.
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
#if 1
  Tensor3 X(dim);
  X.SetGlobalValue(1.0);
  NormalizeStochastic(X);
#else
  Tensor3 X = EmpiricalSecondOrder(seqs);
#endif
  double curr_ll = LogLikelihood(X, seqs);

  int niter = 1000;
  double starting_step_size = 1e-6;
  for (int iter = 0; iter < niter; ++iter) {
    double step_size = starting_step_size / (iter + 1);
    Tensor3 grad = Gradient(seqs, X);
    Tensor3 Y = SRSGradientUpdate(X, step_size, grad);
    double next_ll = LogLikelihood(Y, seqs);
    if (next_ll > curr_ll) {
      X = Y;
      curr_ll = next_ll;
      std::cerr << curr_ll << " " << step_size << std::endl;
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
      int k = 0;  // Always starts at 0
      if (l > 1) {
	k = seq[l - 2];
      }
      int j = seq[l - 1];
      int i = seq[l];
      ll += log(P(i, j, k));
    }
  }
  return ll;
}

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

int main(int argc, char **argv) {
  int N = 5;
  int num_seqs = 10;
  int samples_per_seq = 10000;

  std::vector< std::vector<int> > seqs;
  Tensor3 P = RandomTPT(N);
  Simulate(P, seqs, num_seqs, samples_per_seq);

  double oracle_ll = LogLikelihood(P, seqs);
  Tensor3 PSO = EmpiricalSecondOrder(seqs);
  double empirical_ll = LogLikelihood(PSO, seqs);
  double so_ll = SecondOrderLogLikelihood(PSO, seqs);
  Tensor3 PSRS = EstimateSRS(seqs);
  double srs_ll = LogLikelihood(PSRS, seqs);

  std::cout << "Oracle LL: " << oracle_ll << std::endl;
  std::cout << "Empirical LL: " << empirical_ll << std::endl;
  std::cout << "Second-order LL: "<< so_ll << std::endl;
  std::cout << "SRS LL: "<< srs_ll << std::endl;

  int num_total = (samples_per_seq - 1) * num_seqs;
  double ll_diff1 = exp((empirical_ll - srs_ll) / num_total);
  double ll_diff2 = exp((oracle_ll - srs_ll) / num_total);
  std::cout << ll_diff1 << " " << ll_diff2 << std::endl;
  double P_diff1 = L1Diff(P, PSO) / (N * N);
  double P_diff2 = L1Diff(P, PSRS) / (N * N);
  std::cout << P_diff1 << " " << P_diff2 << std::endl;
}
