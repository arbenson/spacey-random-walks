#!/bin/bash

# Learn the spacey random walk model for the synthetically generated
# trajectories from the random transition probability tensors.

dir=processed_data/synthetic_experiments/uniform

for i in {1..20}; do
    ./learn \
        -n 40000 \
        -s 1 \
        -r 0.5 \
        -m 1e-16 \
        -u 1000 \
	-t $dir/seqs-4-100-200-train.$i.txt \
	-e $dir/seqs-4-100-200-test.$i.txt  \
	-p $dir/P-4-100-200.$i.txt \
	-o $dir/P-4-100-200-learned.$i.txt \
	> $dir/P-4-100-200-results.$i.txt
done
