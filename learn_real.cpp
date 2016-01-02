#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "math.h"

#include "common_srw.hpp"
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
  for (int j = 0; j < N; ++j) {
    for (int k = 0; k < N; ++k) {
      auto col = Y.GetSlice1(j, k);
      double sum = Sum(col);
      if (sum < 1 - 1e-4 || sum > 1 + 1e-4) {
	std::cerr << "Sum: " << sum << std::endl;
	assert(0);
      }
    }
  }
  return Y;
}

Tensor3 EmpiricalSecondOrder(const std::vector< std::vector<int> >& seqs) {
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

Tensor3 EstimateSRW(const std::vector< std::vector<int> >& seqs) {
#if 1
  Tensor3 X = EmpiricalSecondOrder(seqs);
#else
  Tensor3 X(MaximumIndex(seqs) + 1);
  X.SetGlobalValue(1.0);
  NormalizeStochastic(X);
#endif
  double curr_ll = LogLikelihoodTrain(seqs, X);
  double generalization_error = SpaceyRMSE(seqs, X);
  std::cerr << "Starting log likelihood: " << curr_ll << std::endl;
  std::cerr << "Generalization RMSE: " << generalization_error << std::endl;
#if 0
  double srw_pak1 = SpaceyPrecisionAtK(seqs, X, 1);
  double srw_pak3 = SpaceyPrecisionAtK(seqs, X, 3);
  double srw_pak5 = SpaceyPrecisionAtK(seqs, X, 5);
  std::cerr << "Precisions@{1,3,5}: " 
            << srw_pak1 << " "
            << srw_pak3 << " "
            << srw_pak5 << std::endl;
#endif

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
#if 0
      srw_pak1 = SpaceyPrecisionAtK(seqs, Y, 1);
      srw_pak3 = SpaceyPrecisionAtK(seqs, Y, 3);
      srw_pak5 = SpaceyPrecisionAtK(seqs, Y, 5);
      std::cerr << "Precisions@{1,3,5}: "
                << srw_pak1 << " "
                << srw_pak3 << " "
                << srw_pak5 << std::endl;
#endif
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
    double sum = Sum(col);
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
      assert(l > 0);
      int j = seq[l - 1];
      int i = seq[l];
      assert(i < P.dim());
      assert(j < P.dim());
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

void PerPlaceSpaceyErrors(Tensor3& P, std::vector< std::vector<int> >& seqs) {
  std::vector<int> num_correct1(P.dim());  // precision @ 1
  std::vector<int> num_correct2(P.dim());  // precision @ 2
  std::vector<int> num_correct3(P.dim());  // precision @ 3
  std::vector<int> num_total(P.dim());
  std::vector<double> err;

  for (auto& seq : seqs) {
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
      err[i] += val * val;
      num_total[i] += 1;

      std::vector<double> probs(P.dim(), 0.0);
      for (int r = 0; r < occupancy.size(); ++r) {
        for (int ii = 0; ii < P.dim(); ++ii) {
          probs[ii] += occupancy[r] * P(ii, j, r);
        }
      }
      if (InTopK(probs, i, 1)) { num_correct1[i] += 1; }
      if (InTopK(probs, i, 2)) { num_correct2[i] += 1; }
      if (InTopK(probs, i, 3)) { num_correct3[i] += 1; }
    }
  }
  
  // Compute RMSEs
  for (int i = 0; i < err.size(); ++i) {
    err[i] = sqrt(err[i] / num_total[i]);
  }
  
  // Compute Precision@ks
  std::vector<double> precision_at1(P.dim());
  std::vector<double> precision_at2(P.dim());
  std::vector<double> precision_at3(P.dim());
  for (int i = 0; i < P.dim(); ++i) {
    int total = num_total[i];
    precision_at1[i] = static_cast<double>(num_correct1[i]) / total;
    precision_at2[i] = static_cast<double>(num_correct2[i]) / total;
    precision_at3[i] = static_cast<double>(num_correct3[i]) / total;
  }

  for (int i = 0; i < P.dim(); ++i) {
    std::cout << i << " "
              << num_total[i] << " "
              << err[i] << " "
              << precision_at1[i] << " "
              << precision_at2[i] << " "
              << precision_at3[i] << std::endl;
  }
  
}


void PerPlaceSecondOrderErrors(Tensor3& P, std::vector< std::vector<int> >& seqs) {
  std::vector<int> num_correct1(P.dim());  // precision @ 1
  std::vector<int> num_correct2(P.dim());  // precision @ 2
  std::vector<int> num_correct3(P.dim());  // precision @ 3
  std::vector<int> num_total(P.dim());
  std::vector<double> err;

  for (auto& seq : seqs) {
    for (int l = EndingIndex(seq); l < seq.size(); ++l) {
      int i = seq[l];
      int j = seq[l - 1];
      int k = seq[l - 2];
      double val = 1 - P(i, j, k);
      err[i] += val * val;
      num_total[i] += 1;

      auto probs = P.GetSlice1(j, k);
      if (InTopK(probs, i, 1)) { num_correct1[i] += 1; }
      if (InTopK(probs, i, 2)) { num_correct2[i] += 1; }
      if (InTopK(probs, i, 3)) { num_correct3[i] += 1; }
    }
  }
  
  // Compute RMSEs
  for (int i = 0; i < err.size(); ++i) {
    err[i] = sqrt(err[i] / num_total[i]);
  }
  
  // Compute Precision@ks
  std::vector<double> precision_at1(P.dim());
  std::vector<double> precision_at2(P.dim());
  std::vector<double> precision_at3(P.dim());
  for (int i = 0; i < P.dim(); ++i) {
    int total = num_total[i];
    precision_at1[i] = static_cast<double>(num_correct1[i]) / total;
    precision_at2[i] = static_cast<double>(num_correct2[i]) / total;
    precision_at3[i] = static_cast<double>(num_correct3[i]) / total;
  }

  for (int i = 0; i < P.dim(); ++i) {
    std::cout << i << " "
              << num_total[i] << " "
              << err[i] << " "
              << precision_at1[i] << " "
              << precision_at2[i] << " "
              << precision_at3[i] << std::endl;
  }
}


int main(int argc, char **argv) {
  std::vector< std::vector<int> > seqs;
  ReadSequences(argv[1], seqs);

  std::cerr << seqs.size() << " sequences." << std::endl;

  Tensor3 PSO = EmpiricalSecondOrder(seqs);
  Matrix2 PFO = EmpiricalFirstOrder(seqs);

  std::cerr << "First order RMSE: "
            << FirstOrderRMSE(PFO, seqs) << std::endl
            << "Second order RMSE: "
            << SecondOrderRMSE(PSO, seqs) << std::endl
            << "Second order uniform collapse RMSE: "
            << SecondOrderUniformCollapseRMSE(PSO, seqs) << std::endl;

  double so_pak1 = SecondOrderPrecisionAtK(PSO, seqs, 1);
  double so_pak3 = SecondOrderPrecisionAtK(PSO, seqs, 3);
  double so_pak5 = SecondOrderPrecisionAtK(PSO, seqs, 5);
  std::cerr << "SO precisions@{1,3,5}: "
            << so_pak1 << " "
            << so_pak3 << " "
            << so_pak5 << std::endl;

  double fo_pak1 = FirstOrderPrecisionAtK(PFO, seqs, 1);
  double fo_pak3 = FirstOrderPrecisionAtK(PFO, seqs, 3);
  double fo_pak5 = FirstOrderPrecisionAtK(PFO, seqs, 5);
  std::cerr << "FO precisions@{1,3,5}: "
            << fo_pak1 << " "
            << fo_pak3 << " "
            << fo_pak5 << std::endl;

  Tensor3 PSRW = EstimateSRW(seqs);
  int N = PSRW.dim();
  double diff = L1Diff(PSRW, PSO) / (N * N);

  std::cerr << "Difference: " << diff << std::endl;

#if 0
  PerPlaceSpaceyErrors(PSRW, seqs);  
  PerPlaceSecondOrderErrors(PSO, seqs);

  int dimension = PSRW.dim();
  assert(dimension == PSO.dim());
  //for (int j = 0; j < dimension; ++j) {
  int j = 2;
  for (int k = 0; k < dimension; ++k) {
    for (int i = 0; i < dimension; ++i) {
      std::cout << PSRW(i, j, k) << " "
                << PSO(i, j, k) << " "
                << i << " "
                << j << " "
                << k << std::endl;
    }
  }
#endif
}
