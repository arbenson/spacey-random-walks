#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "math.h"


class Tensor3 {
public:
  Tensor3(int N) : N_(N) { data_.resize(N * N * N); }

  double Get(int i, int j, int k) const {
    return data_[k + j * N_ + i * N_ * N_];
  }

  void Set(int i, int j, int k, double val) {
    data_[k + j * N_ + i * N_ * N_] = val;
  }

  // Get T(:, j, k)
  std::vector<double> GetSlice1(int j, int k) const {
    std::vector<double> col(N_);
    for (int i = 0; i < N_; ++i) {
      col[k] = Get(i, j, k);
    }
    return col;
  }

  // Get T(i, j, :)
  std::vector<double> GetSlice3(int i, int j) const {
    std::vector<double> col(N_);
    for (int k = 0; k < N_; ++k) {
      col[k] = Get(i, j, k);
    }
    return col;
  }

  // Set T(:, j, k)
  void SetSlice1(int j, int k, std::vector<double> col) {
    assert(col.size() >= N_);
    for (int i = 0; i < N_; ++i) {
      Set(i, j, k, col[i]);
    }
  }

  // Set T(i, j, :)
  void SetSlice3(int i, int j, std::vector<double> col) {
    assert(col.size() >= N_);
    for (int k = 0; k < N_; ++k) {
      Set(i, j, k, col[k]);
    }
  }

  void SetGlobalValue(double val) {
    for (int i = 0; i < N_; ++i) {
      for (int j = 0; j < N_; ++j) {
	for (int k = 0; k < N_; ++k) {
	  Set(i, j, k, val);
	}
      }
    }
  }

  int dim() const { return N_; }
  

private:
  std::vector<double> data_;
  int N_;
};

template <typename T>
std::vector<double> ScaledVec(const std::vector<T>& vec,
			      double val) {
  std::vector<double> svec(vec.size());
  for (int i = 0; i < vec.size(); ++i) {
    svec[i] = vec[i] * val;
  }
  return svec;
}

void Gradient(std::vector< std::vector<int> >& seqs,
	      Tensor3& X, Tensor3& G) {
  G.SetGlobalValue(0.0);
  for (auto& seq : seqs) {
    std::vector<int> history(X.dim(), 1);
    int total = X.dim();
    for (int l = 1; l < seq.size(); ++l) {
      int i = seq[l];
      int j = seq[l - 1];
      auto occupancy = ScaledVec(history, 1.0 / total);
      std::vector<double> vals(occupancy.size());
      double sum = 0.0;
      for (int k = 0; k < vals.size(); ++k) {
	vals[k] = occupancy[k] * exp(X.Get(i, j, k));
	sum += vals[k];
      }
      for (int k = 0; k < vals.size(); ++k) {
	G.Set(i, j, k, G.Get(i, j, k) + vals[k] / sum);
      }
      history[i] += 1;
      ++total;
    }
  }
}

double LogLikelihood(std::vector< std::vector<int> >& seqs,
		     Tensor3& X) {
  double ll = 0.0;
  for (auto& seq : seqs) {
    std::vector<int> history(X.dim(), 1);
    int total = X.dim();
    for (int l = 1; l < seq.size(); ++l) {
      int i = seq[l];
      int j = seq[l - 1];
      auto occupancy = ScaledVec(history, 1.0 / total);
      double sum = 0.0;
      for (int k = 0; k < occupancy.size(); ++k) {
	sum += occupancy[k] * exp(X.Get(i, j, k));
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
      auto col = P.GetSlice1(j, k);
      double sum = 0.0;
      for (int i = 0; i < col.size(); ++i) {
	col[i] = exp(col[i]);
	sum += col[i];
      }
      for (int i = 0; i < col.size(); ++i) {
	col[i] = log(col[i] / sum);
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

int main(int argc, char **argv) {
  std::vector< std::vector<int> > seqs;
  ReadSequences(argv[0], seqs);
  int dim = MaximumIndex(seqs) + 1;
  Tensor3 X(dim);
  X.SetGlobalValue(-1);
  NormalizeStochastic(X);

  Tensor3 Grad(dim);
  Grad.SetGlobalValue(0);
  Gradient(seqs, X, Grad);
}
