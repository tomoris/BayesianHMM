# Unsupervised POS induction using Bayesian HMM
This is implementation of Bayesian HMM (https://www.aclweb.org/anthology/D10-1056/) using c++.

## Requirement
- C++ 11
- CMake

## Build
```
mkdir src/build
cd src/build
cmake -DUSE_PARALLEL_MODE=TRUE -DCMAKE_BUILD_TYPE=Release ..
make
make install
```
You can get python bindings if you want.
```
git submodule update --init --recursive
mkdir src/build
cd src/build
cmake -DUSE_PARALLEL_MODE=TRUE -DBUILD_PYTHON_MODULE=TRUE -DPYTHON_EXECUTABLE=`which python` -DCMAKE_BUILD_TYPE=Release ..
make
make install
```

## Usage
```
./bin/bhmm --file data/train.txt
```
This is usage of python binding.
```
python bin/main.py --file data/train.txt
```

### Example
Example of "Alice in Wonderland".
- Hyperparameters
  - epoch: 1000 (but we need bigger number for convergence.)
  - others: default parameters

![heatmap](https://user-images.githubusercontent.com/19759196/76062135-658c1a80-5fc8-11ea-8d73-e777f0f47700.png)

## TODO
- add save function
- add hyperparamer inference function
- extend to semi-supervised manner

## License
MIT