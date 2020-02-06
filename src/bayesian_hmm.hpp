#include <vector>
#include <map>

using MyWordIdType = unsigned int;

using MyTagIdType = unsigned int;
const MyTagIdType BEGIN = 0;

const std::string DELIMITER = "-";

class BayesianHMM
{
public:
    BayesianHMM(const int tag_size);
    void Hello();

private:
    const int n_ = 3;
    const int tag_size_;
    int vocab_size_;
    double alpha_;
    double beta_;

    std::map<std::string, int> tag_ngram_count_;
    std::map<std::string, int> word_emission_count_;

    std::string join(const std::vector<MyTagIdType> &v, const std::string delim = "");
    void addNgramParameter(const std::vector<MyTagIdType> &ngram);
    void removeNgramParameter(const std::vector<MyTagIdType> &ngram);
    void addWordEmissionParameter(MyWordIdType word_id, MyTagIdType k);
    void removeWordEmissionParameter(MyWordIdType word_id, MyTagIdType k);
    void samplingTthTag(const int t, const std::vector<MyTagIdType> &tag_sent, const std::vector<MyWordIdType> &word_sent);
    double calcWordProbGivenTag(MyWordIdType word_id, MyTagIdType k);
    double calcTagNgramProb(const std::vector<MyTagIdType> &ngram, const double add_denominator = 0.0, const double add_numerator = 0.0);
    double calcTagPosteriorScore(MyTagIdType k, const int t, const std::vector<MyTagIdType> &tag_sent, MyWordIdType word_id);
};