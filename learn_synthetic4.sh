#!/bin/bash

# Learn the spacey random walk model for the synthetically generated
# trajectories.

dir=processed_data/synthetic_experiments/uniform

for i in {1..20}; do
    ./learn \
	$dir/seqs-4-100-200-train.$i.txt \
	$dir/seqs-4-100-200-test.$i.txt  \
	$dir/P-4-100-200.$i.txt \
	$dir/P-4-100-200-learned.$i.txt \
	> $dir/P-4-100-200-results.$i.txt
done