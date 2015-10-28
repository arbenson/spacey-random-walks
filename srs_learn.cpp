#include <iostream>
#include <math>
#include <sstream>
#include <string>
#include <vector>

class Tensor3 {
public:
  Tensor3(int N) : N_(N) { data_.resize(N * N * N); }

  double Get(int i, int j, int k) {
    return data_[k + j * N + i * N * N];
  }

  double Set(int i, int j, int k, double val) {
    data_[k + j * N + i * N * N] = val;
  }

  // Get T(:, j, k)
  std::vector<double> GetSlice1(int j, int k) {
    std::vector<double> col(N_);
    for (int i = 0; i < N_; ++i) {
      col[k] = Get(i, j, k);
    }
    return col;
  }

  // Get T(:, j, k)
  std::vector<double> GetSliceExp1(int j, int k) {
    std::vector<double> col(N_);
    for (int i = 0; i < N_; ++i) {
      col[k] = exp(Get(i, j, k));
    }
    return col;
  }

  // Get T(i, j, :)
  std::vector<double> GetSlice3(int i, int j) {
    std::vector<double> col(N_);
    for (int k = 0; k < N_; ++k) {
      col[k] = Get(i, j, k);
    }
    return col;
  }

  // Get T(i, j, :)
  std::vector<double> GetSliceExp3(int i, int j) {
    std::vector<double> col(N_);
    for (int k = 0; k < N_; ++k) {
      col[k] = exp(Get(i, j, k));
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

  const int dim() { return N_; }

private:
  std::vector<double> data_;
  int N_;
};

void Gradient(std::vector< std::vector<int> >& seqs,
	      Tensor3& X, Tensor3& G) {
  // ZerOut(G)
  for (auto& seq : seqs) {
    std::vector<int> history(X.dim(), 1);
    int total = X.dim();
    for (int l = 1; l < seq.size(); ++l) {
      i = seq[l];
      j = seq[l - 1];
      auto occupancy = ScaledVec(history, 1.0 / total);
      // vals = occ_v * np.exp(X[i, j, :])        
      // G[i, j, :] += vals / np.sum(vals)
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
      i = seq[l];
      j = seq[l - 1];
      auto occupancy = ScaledVec(history, 1.0 / total);
      // occ_v = occ_s / np.sum(occ_s)
      // trans = np.exp(X[i, j, :])
      // ll += np.log(np.sum(occ_v * trans))
      history[i] += 1;
      ++total;
    }
  }
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

void ReadSequences(std::string filename) {
  std::vector< std::vector<int> > seqs;
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



