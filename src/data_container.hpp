
#include "common.hpp"

#include <map>
#include <string>
#include <vector>

class DataContainer
{
public:
    DataContainer(const std::string file_name, const std::string delimiter = " ");

    std::vector<std::vector<MyWordIdType>> corpus;
    std::vector<std::vector<MyTagIdType>> tag_corpus;

    MyWordIdType GetWordVocabSize() const;

private:
    const int ngram_size_ = NGRAM_SIZE;

    std::map<std::string, MyWordIdType> w2id_;
    std::map<MyWordIdType, std::string> id2w_;
    MyWordIdType vocab_size_;

    void load(const std::string file_name, const std::string delimiter);
    std::vector<std::string> split(const std::string str, const std::string delimiter) const;
};