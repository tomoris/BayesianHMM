
#include "common.hpp"

#include <map>
#include <random>
#include <string>
#include <vector>

const std::string DELIMITER = "-";

class BayesianHMM
{
public:
    BayesianHMM(const int tag_size, const MyWordIdType vocab_size, const double alpha, const double beta);
    void Train(const std::vector<std::vector<MyWordIdType>> &corpus, std::vector<std::vector<MyTagIdType>> &tag_corpus, const int epoch, const int max_threads = 1);
    void Test(const std::vector<std::vector<MyWordIdType>> &corpus, std::vector<std::vector<MyTagIdType>> &tag_corpus, const int max_threads = 1);
    std::vector<std::vector<MyTagIdType>> TrainForPython(const std::vector<std::vector<MyWordIdType>> &corpus, std::vector<std::vector<MyTagIdType>> &tag_corpus, const int epoch, const int max_threads = 1);
    std::vector<std::vector<MyTagIdType>> TestForPython(const std::vector<std::vector<MyWordIdType>> &corpus, std::vector<std::vector<MyTagIdType>> &tag_corpus, const int max_threads = 1);
    double CalcTagScoreGivenCorpus(const std::vector<std::vector<MyWordIdType>> &corpus, const std::vector<std::vector<MyTagIdType>> &tag_corpus, const int max_threads = 1) const;

private:
    const int n_ = NGRAM_SIZE;
    const int tag_size_;
    int vocab_size_;
    double alpha_;
    double beta_;

    std::map<std::string, int> tag_ngram_count_;
    std::map<std::string, int> word_emission_count_;

    std::mt19937 random_generator_;

    std::string join(const std::vector<MyTagIdType> &v, const std::string delim = "") const;
    void addNgramParameter(const std::vector<MyTagIdType> &ngram, const int recursive);
    void removeNgramParameter(const std::vector<MyTagIdType> &ngram, const int recursive);
    void addWordEmissionParameter(const MyWordIdType word_id, const MyTagIdType k);
    void removeWordEmissionParameter(const MyWordIdType word_id, const MyTagIdType k);
    MyTagIdType samplingTthTag(const int t, const std::vector<MyTagIdType> &tag_sent, const MyWordIdType word_id, const int max_threads, const bool sampling=true);
    void gibbsSamplingTthTag(const int t, std::vector<MyTagIdType> &tag_sent, const std::vector<MyWordIdType> &word_sent, const int max_threads);
    void gibbsSamplingAtSent(std::vector<MyTagIdType> &tag_sent, const std::vector<MyWordIdType> &word_sent, const int max_threads);
    double calcWordProbGivenTag(const MyWordIdType word_id, const MyTagIdType k) const;
    double calcTagNgramProb(const std::vector<MyTagIdType> &ngram, const double add_denominator = 0.0, const double add_numerator = 0.0) const;
    double calcTagPosteriorScore(const MyTagIdType k, const int t, const std::vector<MyTagIdType> &tag_sent, const MyWordIdType word_id, const bool sampling) const;
    void initialize(const std::vector<std::vector<MyWordIdType>> &corpus, std::vector<std::vector<MyTagIdType>> &tag_corpus);
};