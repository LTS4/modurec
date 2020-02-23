# Addressing the concept drift for recommender systems

The code includes three bash scripts for each of the datasets used:

* ml-100k.sh : for Movielens 100K
* ml-1m.sh : for Movielens 1M
* ml-10m.sh : for Movielens 10M

The scripts have by default the hyperparameters values used in our experiments and trains/evaluates de Autorec_DT model.

To train the other models, just change the --model parameter from this list of options:

* Autorec_D
* Autorec_DT
* Autorec_DGT