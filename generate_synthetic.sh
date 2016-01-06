#!/bin/bash

# Generate the synthetic trajectories.

# Tensors generated uniformly at random
dir=processed_data/synthetic_experiments/uniform
mkdir -p $dir
for i in {1..20}; do ./sim -d 4 -n 100 -s 200 -o $dir/seqs-4-100-200.$i.txt -t $dir/P-4-100-200.$i.txt; done
for i in {1..20}; do head -80 $dir/seqs-4-100-200.$i.txt > $dir/seqs-4-100-200-train.$i.txt; done
for i in {1..20}; do tail -20 $dir/seqs-4-100-200.$i.txt > $dir/seqs-4-100-200-test.$i.txt; done

# R1
dir=processed_data/synthetic_experiments/R1
mkdir -p $dir
./sim -n 100 -s 200 -o $dir/seqs-R1-100-200.txt -t $dir/R1.txt
head -80 $dir/seqs-R1-100-200.txt > $dir/seqs-R1-100-200-train.txt
tail -20 $dir/seqs-R1-100-200.txt > $dir/seqs-R1-100-200-test.txt

# R2
dir=processed_data/synthetic_experiments/R2
mkdir -p $dir
./sim -n 100 -s 200 -o $dir/seqs-R2-100-200.txt -t $dir/R2.txt
head -80 $dir/seqs-R2-100-200.txt > $dir/seqs-R2-100-200-train.txt
tail -20 $dir/seqs-R2-100-200.txt > $dir/seqs-R2-100-200-test.txt
