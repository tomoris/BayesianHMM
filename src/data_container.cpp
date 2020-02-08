
#include "data_container.hpp"

#include <fstream>
#include <iostream>

DataContainer::DataContainer(const std::string file_name, const std::string delimiter)
{
    vocab_size_ = static_cast<MyWordIdType>(SPECIAL_WORD_SIZE);
    this->load(file_name, delimiter);
}

void DataContainer::load(const std::string file_name, const std::string delimiter)
{
    std::ifstream ifs(file_name);
    std::string str;
    if (ifs.fail())
    {
        std::cerr << "Failed to open file." << std::endl;
        exit(1);
    }

    while (getline(ifs, str))
    {
        auto tokens = this->split(str, delimiter);
        std::vector<MyWordIdType> token_ids(ngram_size_ - 1, BEGIN_WORD_ID);
        std::vector<MyTagIdType> tag_sent(ngram_size_ - 1, BEGIN_TAG_ID);
        for (auto token : tokens)
        {
            auto itr = w2id_.find(token);
            if (itr == w2id_.end())
            {
                w2id_.emplace(token, vocab_size_);
                id2w_.emplace(vocab_size_, token);
                vocab_size_ += 1;
            }
            MyWordIdType token_id = w2id_[token];
            token_ids.push_back(token_id);
            tag_sent.push_back(SPECIAL_TAG_SIZE);
        }
        for (int n = 0; n < ngram_size_ - 1; n++)
        {
            token_ids.push_back(END_WORD_ID);
            tag_sent.push_back(END_TAG_ID);
        }
        corpus.push_back(token_ids);
        tag_corpus.push_back(tag_sent);
    }
}

std::vector<std::string> DataContainer::split(const std::string str, const std::string delimiter) const
{
    if (delimiter == "")
    {
        return {str};
    }
    std::vector<std::string> result;
    std::string tstr = str + delimiter;
    long l = tstr.length(), sl = delimiter.length();
    std::string::size_type pos = 0, prev = 0;

    for (; pos < static_cast<std::string::size_type>(l) && (pos = tstr.find(delimiter, pos)) != std::string::npos; prev = (pos += sl))
    {
        result.emplace_back(tstr, prev, pos - prev);
    }
    return result;
}

MyWordIdType DataContainer::GetWordVocabSize() const
{
    return vocab_size_;
}