#!/bin/bash

# Learn the spacey random walk model for the synthetically generated
# trajectories from the R_1 and R_2 transition probability tesnors.  Need to run
# 'make learn' first.


dir=processed_data/synthetic_experiments/R1
./learn -n 40000 -s 1 -r 0.5 -m 1e-16 -u 1000 \
    -t $dir/seqs-R1-100-200-train.txt -e $dir/seqs-R1-100-200-test.txt  \
    -p $dir/R1.txt -o $dir/R1-learned.txt \
    > $dir/R1-results.txt

dir=processed_data/synthetic_experiments/R2
./learn -n 40000 -s 1 -r 0.5 -m 1e-16 -u 1000 \
    -t $dir/seqs-R2-100-200-train.txt -e $dir/seqs-R2-100-200-test.txt  \
    -p $dir/R2.txt -o $dir/R2-learned.txt \
    > $dir/R2-results.txt
