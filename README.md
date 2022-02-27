# Auto-weighted Robust Federated Learning with Corrupted Data Sources

This repository contains the code and experiments for the paper:

[Auto-weighted Robust Federated Learning with Corrupted Data Sources](https://arxiv.org/abs/2101.05880)

## Datasets

1. CIFAR-10
  * **Overview:** Image Dataset. See [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
  * **Details:** 10 different classes, images are 32 by 32 pixels.
  * **Task:** Image Classification

2. FEMNIST

  * **Overview:** Image Dataset
  * **Details:** 62 different classes (10 digits, 26 lowercase, 26 uppercase), images are 28 by 28 pixels (with option to make them all 128 by 128 pixels), 3500 users
  * **Task:** Image Classification

3. Shakespeare

  * **Overview:** Text Dataset of Shakespeare Dialogues
  * **Details:** 1129 users (reduced to 660 with our choice of sequence length.
  * **Task:** Next-Character Prediction

## Notes

- Install the libraries listed in ```requirements.txt```
    - I.e. with pip: run ```pip3 install -r requirements.txt```
    - To prepare the dataset for the paper, run  ```sudo configure.sh```
- Go to directory of respective dataset for instructions on generating data
- ```models``` directory contains instructions on running baseline reference implementations

## Ref

### LEAF benchmark
* **Homepage:** [leaf.cmu.edu](https://leaf.cmu.edu)
* **Paper:** ["LEAF: A Benchmark for Federated Settings"](https://arxiv.org/abs/1812.01097)
