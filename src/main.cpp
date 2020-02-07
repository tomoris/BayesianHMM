
#include "bayesian_hmm.hpp"
#include "data_container.hpp"

#include <iostream>
#include <vector>

int main()
{
    // std::vector<std::vector<MyWordIdType>> corpus = {{1, 2, 3}, {4, 2, 3}};

    DataContainer data_container("../data/train.txt", " ");

    const int pos_size = 10;
    const int epoch = 10;

    BayesianHMM hmm(pos_size, data_container.GetWordVocabSize());
    hmm.Train(data_container.corpus, data_container.tag_corpus, epoch);
    hmm.Hello();
}