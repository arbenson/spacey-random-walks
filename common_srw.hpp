#ifndef _COMMON_SRW_HPP_
#define _COMMON_SRW_HPP_

#include <cmath>
#include <vector>

#include "tensor3.hpp"


// Return sum of a vector
template <typename T>
T Sum(const std::vector<T>& vec) {
  T sum = 0;
  for (T v : vec) {
    sum += v;
  }
  return sum;
}

// Return the L1 norm of the vector
template <typename T>
T L1Norm(const std::vector<T>& vec) {
  T sum = 0;
  for (T v : vec) {
    sum += std::abs(v);
  }
  return sum;
}  

// Return the L1-normalized vector.  Leaves the 0 vector as is.
template <typename T>
std::vector<double> Normalized(const std::vector<T>& vec) {
  std::vector<double> nvec(vec.size());
  double sum = L1Norm(vec);
  if (sum > 0.0) {
    for (int i = 0; i < nvec.size(); ++i) {
      nvec[i] = static_cast<double>(vec[i]) / sum;
    }
  }
  return nvec;
}

// Given a vector of probabilities (vec) and an index in {0, 1, ...,
// vec.size()-1}, determine if the vector value at the index is one of the top K
// largest values.
bool InTopK(const std::vector<double>& vec, int index, int K);

// Maximum index used for all states (starts at 0).
int MaximumIndex(const std::vector< std::vector<int> >& seqs);

// Minimum l2 projection onto the simplex.
std::vector<double> SimplexProjection(const std::vector<double>& vec);

// Normalize columns of a transition probability tensor to be stochastic.
void NormalizeStochastic(Tensor3& P);

// Use the minimum l2 projection to project each column of the tensor onto the
// simplex.
void ProjectColumnsOntoSimplex(Tensor3& Y);

// Sample from a discrete probability distribution.
int Choice(const std::vector<double>& probs);

// || vec(P1) - vec(P2) ||_1
double L1Diff(const Tensor3& P1, const Tensor3& P2);

// || v1 - v2 ||_1
double L1Diff(const std::vector<double>& v1, const std::vector<double>& v2);

// Apply length-N^2 vector x to N^2 x N^2 transition matrix represented by the
// N x N x N transition probability tensor P.
std::vector<double> Apply(const Tensor3& P, const std::vector<double>& x);

// Compute y = P x^2 (also y = R (x \kron x))
std::vector<double> TensorApply(const Tensor3& P, const std::vector<double>& x);

// Compute stationary distribution of the second-order Markov chain represented
// by P.
std::vector<double> Stationary(const Tensor3& P);

// Compute the marginals of the stationary 
std::vector<double> StationaryMarginals(const Tensor3& P);

// Simple shifted power method to compute the spacey stationary distribution.
std::vector<double> SpaceyStationary(const Tensor3& P,
				     int max_iter /* =1000 */,
				     double gamma /* =0.01 */,
				     double tol   /* =1e-12 */);

// Read sequences from filename and store the resuts in seqs.
void ReadSequences(std::string filename,
                   std::vector< std::vector<int> >& seqs);

// Write the tensor to file with the format
//    i j k P(i, j, k)
void WriteTensor(const Tensor3& P, const std::string& outfile);

#endif  // _COMMON_SRW_HPP_
