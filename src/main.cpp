
#include "bayesian_hmm.hpp"
#include "data_container.hpp"

#include "cxxopts.hpp"

#include <iostream>
#include <vector>

cxxopts::ParseResult parse(int argc, char *argv[])
{
    try
    {
        cxxopts::Options options(argv[0], "Bayesian HMM implementation");
        options.add_option("", "", "help", "Print help", cxxopts::value<bool>(),
                           "");
        options.add_option("", "", "file", "training text file path (required)",
                           cxxopts::value<std::string>(), "file path");
        options.add_option("", "", "tag", "tag size",
                           cxxopts::value<int>()->default_value("10"), "num");
        options.add_option("", "e", "epoch", "epoch",
                           cxxopts::value<int>()->default_value("100"), "num");
        options.add_option("", "", "alpha",
                           "hyperparameter (0.0 < alpha, it is better that alpha "
                           "is lower than 1.0)",
                           cxxopts::value<double>()->default_value("0.1"), "num");
        options.add_option(
            "", "", "beta",
            "hyperparameter (0.0 < beta, it is better that beta is lower than 1.0)",
            cxxopts::value<double>()->default_value("0.1"), "num");
        options.add_option("", "", "threads", "number of  maximum threads",
                           cxxopts::value<int>()->default_value("1"), "num");

        auto result = options.parse(argc, argv);

        if (result.count("help") || !result.count("file") ||
            result["alpha"].as<double>() <= 0.0 ||
            result["beta"].as<double>() <= 0.0)
        {
            std::cout << options.help() << std::endl;
            exit(0);
        }

        return result;
    }
    catch (const std::exception &e)
    {
        std::cerr << "command line parse error" << '\n';
        std::cerr << e.what() << '\n';
        exit(1);
    }
}

int main(int argc, char *argv[])
{
    auto cmd = parse(argc, argv);
    std::string file_name = cmd["file"].as<std::string>();
    const int epoch = cmd["epoch"].as<int>();
    const int pos_size = cmd["tag"].as<int>();
    const double alpha = cmd["alpha"].as<double>();
    const double beta = cmd["beta"].as<double>();
    const int max_threads = cmd["threads"].as<int>();

    std::cout << "loading input file" << std::endl;
    DataContainer data_container(file_name, " ");

    std::cout << "building model" << std::endl;
    std::cout << data_container.tag_corpus[0][3] << std::endl;
    BayesianHMM hmm(pos_size, data_container.GetWordVocabSize(), alpha, beta);

    std::cout << "training" << std::endl;
    hmm.Train(data_container.corpus, data_container.tag_corpus, epoch,
              max_threads);
    std::cout << data_container.tag_corpus[0][3] << std::endl;
    std::cout << "end training" << std::endl;
}