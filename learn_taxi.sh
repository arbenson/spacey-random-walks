#!/bin/bash

# Learn the spacey random walk model for the taxi data.  Need to run
# 'make learn' first.


dir=processed_data/taxi

./learn \
    -n 200 \
    -s 3.125e-08 \
    -r 0.5 \
    -m 1e-16 \
    -u 5 \
    -t $dir/manhattan-year-train.txt \
    -e $dir/manhattan-year-test.txt \
    -o $dir/manhattan-year-learned.txt \
    > $dir/manhattan-year-results.txt
