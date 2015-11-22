#include <iostream>
#include <random>

#include "common_srs.hpp"
#include "tensor3.hpp"

Tensor3 EmpiricalSecondOrder(std::vector< std::vector<int> >& seqs) {
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

double LogLikelihood(std::vector< std::vector<int> >& seqs,
		     Tensor3& P) {
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



int main(int argc, char **argv) {
  int N = 4;
  int num_seqs = 20;
  int samples_per_seq = 500;

  std::vector< std::vector<int> > seqs;
  Tensor3 P = RandomTPT(N);
  Simulate(P, seqs, num_seqs, samples_per_seq);
}
