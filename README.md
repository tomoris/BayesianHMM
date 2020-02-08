# Unsupervised POS induction using Bayesian HMM

This is implementation of Bayesian HMM (https://www.aclweb.org/anthology/D10-1056/) using c++.

## Requirement

- C++ 11
- CMake

## Build
```
mkdir src/build
cd src/build
cmake ..
make
```

## Usage
```
./src/build/bhmm --file data/train.txt
```

## TODO

- add save function
- add python binding
- add hyperparamer inference function
- extend to semi-supervised manner