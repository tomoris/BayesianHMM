#include <iostream>
#include <vector>

#include "bayesian_hmm.hpp"

int main()
{
    std::vector<std::vector<MyWordIdType>> corpus = {{1, 2, 3}, {4, 2, 3}};

    BayesianHMM hmm(10);
    hmm.Hello();
}