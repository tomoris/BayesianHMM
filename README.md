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
make install
```
You can get python bindings if you want.
```
cmake -DBUILD_PYTHON_MODULE=TRUE -DPYTHON_EXECUTABLE=`which python` ..
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

## TODO
- add save function
- add hyperparamer inference function
- extend to semi-supervised manner

## License
MIT