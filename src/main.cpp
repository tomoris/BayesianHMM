
#include "bayesian_hmm.hpp"
#include "data_container.hpp"

#include "cxxopts.hpp"

#include <iostream>
#include <vector>

cxxopts::ParseResult
parse(int argc, char *argv[])
{
    try
    {
        cxxopts::Options options(argv[0], " - example command line options");
        options.add_option("", "", "help", "Print help", cxxopts::value<bool>(), "");
        options.add_option("", "", "file", "training text file path (required)", cxxopts::value<std::string>(), "file path");
        options.add_option("", "", "pos", "pos size", cxxopts::value<int>()->default_value("10"), "num");
        options.add_option("", "e", "epoch", "epoch", cxxopts::value<int>()->default_value("100"), "num");

        auto result = options.parse(argc, argv);

        if (result.count("help") || !result.count("file"))
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
    const int pos_size = cmd["pos"].as<int>();

    std::cout << "loading input file" << std::endl;
    DataContainer data_container(file_name, " ");

    std::cout << "building model" << std::endl;
    BayesianHMM hmm(pos_size, data_container.GetWordVocabSize());

    std::cout << "training" << std::endl;
    hmm.Train(data_container.corpus, data_container.tag_corpus, epoch);
    std::cout << "end training" << std::endl;
    hmm.Hello();
}