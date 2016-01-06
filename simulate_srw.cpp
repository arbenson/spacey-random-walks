#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <getopt.h>

#include "common_srw.hpp"
#include "tensor3.hpp"

static int simulate_R1 = 0;
static int simulate_R2 = 0;
static int simulate_random = 0;
static int problem_dimension = 4;
static int number_of_simulated_sequences = 25;
static int size_of_simulated_sequence = 1000;
static std::string sequence_output_file = "seqs.out";
static std::string tensor_output_file = "P.out";

// Form random transition probability tensor.  Each column is selected uniformly
// at random from the simplex.
Tensor3 UniformRandomTPT(int dimension) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0, 1);
  Tensor3 P(dimension);
  for (int j = 0; j < dimension; ++j) {
    for (int k = 0; k < dimension; ++k) {
      std::vector<double> samples = {0.0, 1.0};
      // Generate dimension - 1 uniform [0, 1] random variables
      for (int l = 0; l < dimension - 1; ++l) {
	samples.push_back(dis(gen));
      }
      std::sort(samples.begin(), samples.end());
      for (int i = 0; i < dimension; ++i) {
	P(i, j, k) = samples[i + 1] - samples[i];
      }
    }
  }
  return P;
}

// Form random transition probability tensor.
Tensor3 LogNormalRandomTPT(int dimension) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::lognormal_distribution<double> dis(0.0,1.0);
  Tensor3 P(dimension);
  for (int i = 0; i < dimension; ++i) {
    for (int j = 0; j < dimension; ++j) {
      for (int k = 0; k < dimension; ++k) {
	P(i, j, k) = dis(gen);
      }
    }
  }
  NormalizeStochastic(P);
  return P;
}

// Transitions corresponding to R_{1} in multilinear pagerank paper.
Tensor3 R1() {
  Tensor3 X(4);
  X.SetGlobalValue(0.0);
  
  // First panel
  X(3, 0, 0) = 1;
  X(3, 1, 0) = 1;
  X(3, 2, 0) = 1;
  X(3, 3, 0) = 1;

  // Second panel
  X(3, 0, 1) = 1;
  X(1, 1, 1) = 1;
  X(2, 2, 1) = 1;
  X(1, 3, 1) = 1;

  // Third panel
  X(3, 0, 2) = 1;
  X(1, 1, 2) = 1;
  X(2, 1, 2) = 1;
  X(2, 2, 2) = 1;
  X(3, 3, 2) = 1;

  // Fourth panel
  X(0, 0, 3) = 1;
  X(3, 0, 3) = 1;
  X(1, 1, 3) = 1;
  X(3, 2, 3) = 1;
  X(0, 3, 3) = 1;

  NormalizeStochastic(X);
  return X;
}

// Transitions corresponding to R_{4,9} in multilinear pagerank paper.
Tensor3 R2() {
  Tensor3 X(4);
  X.SetGlobalValue(0.0);
  
  // First panel
  X(3, 0, 0) = 1;
  X(3, 1, 0) = 1;
  X(3, 2, 0) = 1;
  X(3, 3, 0) = 1;

  // Second panel
  X(3, 0, 1) = 1;
  X(1, 1, 1) = 1;
  X(2, 2, 1) = 1;
  X(1, 3, 1) = 1;

  // Third panel
  X(3, 0, 2) = 1;
  X(1, 1, 2) = 1;
  X(2, 1, 2) = 1;
  X(2, 2, 2) = 1;
  X(0, 3, 2) = 1;

  // Fourth panel
  X(0, 0, 3) = 1;
  X(1, 1, 3) = 1;
  X(0, 2, 3) = 1;
  X(3, 3, 3) = 1;

  NormalizeStochastic(X);
  return X;
}

void Simulate(const Tensor3& P, std::vector< std::vector<int> >& seqs,
              int num_seqs, int num_samples) {
  int dim = P.dim();
  seqs.clear();
  for (int seq_ind = 0; seq_ind < num_seqs; ++seq_ind) {
    std::vector<int> history(dim, 1);
    std::vector<double> occupancy = Normalized(history);
    int j = 0;  // Starts at zero by default
    std::vector<int> seq(num_samples);
    for (int sample_ind = 0; sample_ind < num_samples; ++sample_ind) {
      // Choose from history
      occupancy = Normalized(history);
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

void HandleOptions(int argc, char **argv) {
  static struct option long_options[] =
    {
      {"sim_R1",          no_argument, &simulate_R1, 1},
      {"sim_R2",          no_argument, &simulate_R2, 1},
      {"sim_rand",        no_argument, &simulate_random, 1},
      {"dimension",       required_argument, 0, 'd'},
      {"num_sequences",   required_argument, 0, 'n'},
      {"sequence",        required_argument, 0, 's'},
      {"sequence_output", required_argument, 0, 'o'},
      {"tensor_output",   required_argument, 0, 't'},
      {0, 0, 0, 0}
    };

  int c;
  while (1) {
    int option_index = 0;
    c = getopt_long (argc, argv, "d:n:s:o:t:",
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
    case 'd':
      problem_dimension = atoi(optarg);
      break;
    case 'n':
      number_of_simulated_sequences = atoi(optarg);
      break;
    case 's':
      size_of_simulated_sequence = atoi(optarg);
      break;
    case 'o':
      sequence_output_file = std::string(optarg);
      break;
    case 't':
      tensor_output_file = std::string(optarg);
      break;
    default:
      abort();
    }
  }
}

void WriteSequences(const std::vector< std::vector<int> >& seqs,
		    const std::string& outfile) {
  std::ofstream out;
  out.open(outfile);
  for (const auto& seq : seqs) {
    for (int i = 0; i < seq.size(); ++i) {
      out << seq[i];
      if (i != seq.size() - 1) {
	out << ",";
      }
    }
    out << std::endl;
  }
  out.close();
}

void SimulateR1() {
  std::vector< std::vector<int> > seqs;
  Tensor3 P = R1();
  Simulate(P, seqs, number_of_simulated_sequences, size_of_simulated_sequence);
  WriteTensor(P, tensor_output_file);
  WriteSequences(seqs, sequence_output_file);
}

void SimulateR2() {
  std::vector< std::vector<int> > seqs;
  Tensor3 P = R2();
  Simulate(P, seqs, number_of_simulated_sequences, size_of_simulated_sequence);
  WriteTensor(P, tensor_output_file);
  WriteSequences(seqs, sequence_output_file);
}

int main(int argc, char **argv) {
  HandleOptions(argc, argv);
  if (simulate_R1) {
    SimulateR1();
  } else if (simulate_R2) {
    SimulateR2();
  } else if (simulate_random) {
    Tensor3 P = UniformRandomTPT(problem_dimension);
    std::vector< std::vector<int> > seqs;
    Simulate(P, seqs, number_of_simulated_sequences, size_of_simulated_sequence);
    WriteTensor(P, tensor_output_file);
    WriteSequences(seqs, sequence_output_file);
  }
}
