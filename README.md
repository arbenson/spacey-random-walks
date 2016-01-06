Spacey Random Walker
--------
Austin R. Benson, David F. Gleich, and Lek-Heng Lim

This repository contains research code to accompany our paper.


Synthetic data
--------
The synthetic data used in the numerical experiments is also included in the repository in `processed_data/synthetic_experiments/`.
The data can also be generated again:

      make sim
      bash generate_synthetic.sh

To train and test the spacey random walk model on the data, run:

      make learn
      bash learn_synthetic_R1_R2.sh
      bash learn_synthetic_random.sh

Taxi data
--------
The processed taxi trajectory data used in the numerical experiments is provided in the file `processed_data/taxi/manhattan-year-seqs.txt`.
The raw data can be downloaded [here](http://www.andresmh.com/nyctaxitrips/).
The code used to process this data is in `scripts/form_taxi_trajectories.py`.

To train and test the spacey random walk model on the data, run:

      make learn
      bash learn_taxi.sh
