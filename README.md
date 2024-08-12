# Modurec: Recommender systems with feature and time modulation

This is the source code to reproduce the experiments of the paper "[Modurec: Recommender systems with feature and time modulation](https://ieeexplore.ieee.org/abstract/document/9413676)" by Javier Maroto, Clément Vignac and Pascal Frossard.

## Usage

The code includes three bash scripts for each of the datasets used:

* ml-100k.sh : for Movielens 100K
* ml-1m.sh : for Movielens 1M
* ml-10m.sh : for Movielens 10M

The scripts have by default the hyperparameters values used in our experiments and trains/evaluates de Autorec_DT model.

To train the other models, just change the --model parameter from this list of options:

* Autorec_D
* Autorec_DT
* Autorec_DGT

## Reference
If you find this code useful, please cite the following paper:
```bibtex
@inproceedings{maroto2021modurec,
  title={Modurec: Recommender systems with feature and time modulation},
  author={Maroto, Javier and Vignac, Cl{\'e}ment and Frossard, Pascal},
  booktitle={ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={3615--3619},
  year={2021},
  organization={IEEE}
}

```
