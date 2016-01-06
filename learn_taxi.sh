#!/bin/bash

# Learn the spacey random walk model for the taxi data.

dir=processed_data/taxi

./learn_synthetic \
    $dir/manhattan-year-train.txt \
    $dir/manhattan-year-test.txt \
    $dir/P \
    $dir/manhattan-year-learned.txt \
    > $dir/manhattan-year-results.txt

