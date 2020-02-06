
#include "bayesian_hmm.hpp"

#include <iostream>
#include <cassert>

BayesianHMM::BayesianHMM(const int tag_size) : tag_size_(tag_size)
{
    vocab_size_ = 100;
    alpha_ = 0.1;
    beta_ = 0.1;
}

void BayesianHMM::Hello()
{
    std::cout << "hello" << std::endl;
    std::vector<MyTagIdType> tmp{1, 2, 3};
    this->addNgramParameter(tmp);
    this->addWordEmissionParameter(1, 3);
    double score1 = this->calcWordProbGivenTag(1, 3);
    double score2 = this->calcTagNgramProb(tmp);
}

std::string BayesianHMM::join(const std::vector<MyWordIdType> &v, const std::string delim)
{
    std::string s;
    if (!v.empty())
    {
        s += std::to_string(v[0]);
        for (decltype(v.size()) i = 1, c = v.size(); i < c; ++i)
        {
            s += delim;
            s += std::to_string(v[i]);
        }
    }
    return s;
}

// ここはスムージングしているのか不明
void BayesianHMM::addNgramParameter(const std::vector<MyTagIdType> &ngram)
{
    assert(ngram.size() == n_);

    for (int i = 0; i < n_; i++)
    {
        std::vector<MyTagIdType> tmp(n_ - i);
        copy(ngram.begin() + i, ngram.begin() + n_, tmp.begin());
        std::string ngram_string = this->join(tmp, DELIMITER);
        tag_ngram_count_.insert(std::make_pair(ngram_string, 0));
        tag_ngram_count_[ngram_string] += 1;
        std::cout << ngram_string << " " << tag_ngram_count_[ngram_string] << "   ";
    }
    std::cout << std::endl;
}

// ここはスムージングしているのか不明
void BayesianHMM::removeNgramParameter(const std::vector<MyTagIdType> &ngram)
{
    assert(ngram.size() == n_);

    for (int i = 0; i < n_; i++)
    {
        std::vector<MyTagIdType> tmp(n_ - i);
        copy(ngram.begin() + i, ngram.begin() + n_, tmp.begin());
        std::string ngram_string = this->join(tmp, DELIMITER);
        tag_ngram_count_[ngram_string] -= 1;
        if (tag_ngram_count_[ngram_string] == 0)
        {
            tag_ngram_count_.erase(ngram_string);
        }
    }
}

void BayesianHMM::addWordEmissionParameter(MyWordIdType word_id, MyTagIdType k)
{
    std::string word_emission_string = std::to_string(k) + DELIMITER + std::to_string(word_id);
    word_emission_count_.insert(std::make_pair(word_emission_string, 0));
    word_emission_count_[word_emission_string] += 1;
}

void BayesianHMM::removeWordEmissionParameter(MyWordIdType word_id, MyTagIdType k)
{
    std::string word_emission_string = std::to_string(k) + DELIMITER + std::to_string(word_id);
    word_emission_count_[word_emission_string] -= 1;
    if (word_emission_count_[word_emission_string] == 0)
    {
        word_emission_count_.erase(word_emission_string);
    }
}

// 実際のt+2番目のタグをサンプリング(BEGINの分を2つ文頭に足している,trigramの場合)
void BayesianHMM::samplingTthTag(const int t, const std::vector<MyTagIdType> &tag_sent, const std::vector<MyWordIdType> &word_sent)
{
    // t-th tag に関連するパラメータの削除
    // (tag_{t-2}, tag_{t-1}, tag_{t}), (tag_{t-1}, tag_{t}, tag_{t+1}), (tag_{t}, tag_{t+1}, tag_{t+2})
    for (int i = 0; i < n_; i++)
    {
        if ((t + (n_ - 1) + i) >= tag_sent.size())
        {
            break;
        }
        std::vector<MyTagIdType> ngram;
        copy(tag_sent.begin() + t + i, tag_sent.begin() + t + (n_ - 1) + i, ngram.begin());
        this->removeNgramParameter(ngram);
    }

    // BayesianHMM::removeWordEmissionParameter(t, )

    std::vector<double> scores(tag_size_, 0.0);
    for (int k = 0; k < tag_size_; k++)
    {
        scores[k] = this->calcTagPosteriorScore(static_cast<MyTagIdType>(k), t, tag_sent, word_sent[t + (n_ - 1)]);
    }
}

double BayesianHMM::calcWordProbGivenTag(MyWordIdType word_id, MyTagIdType k)
{
    double k_count = 0.0;
    std::map<std::string, int>::iterator itr = tag_ngram_count_.find(std::to_string(k));
    if (itr != tag_ngram_count_.end())
    {
        k_count = static_cast<double>(itr->second);
    }

    double w_count = 0.0;
    std::string word_emission_string = std::to_string(k) + DELIMITER + std::to_string(word_id);
    itr = word_emission_count_.find(word_emission_string);
    if (itr != word_emission_count_.end())
    {
        w_count = static_cast<double>(itr->second);
    }
    assert(k_count >= w_count);

    double p;
    p = (w_count + beta_) / (k_count + static_cast<double>(vocab_size_) * beta_);

    return p;
}

double BayesianHMM::calcTagNgramProb(const std::vector<MyTagIdType> &ngram, const double add_denominator, const double add_numerator)
{
    assert(ngram.size() == n_);
    assert(add_denominator >= 0.0);
    assert(add_numerator >= 0.0);
    assert(add_denominator >= add_numerator);

    double ngram_count = 0.0;
    std::string ngram_string = this->join(ngram, DELIMITER);
    std::map<std::string, int>::iterator itr = tag_ngram_count_.find(ngram_string);
    if (itr != tag_ngram_count_.end())
    {
        ngram_count = static_cast<double>(itr->second);
    }

    double total_count = 0.0;
    std::vector<MyTagIdType> n_1gram(n_ - 1);
    copy(ngram.begin() + 1, ngram.end(), n_1gram.begin());
    ngram_string = this->join(n_1gram, DELIMITER);
    itr = tag_ngram_count_.find(ngram_string);
    if (itr != tag_ngram_count_.end())
    {
        total_count = static_cast<double>(itr->second);
    }
    assert(total_count >= ngram_count);

    double p;
    p = (ngram_count + add_numerator + alpha_) / (total_count + add_denominator + tag_size_ * alpha_);
    return p;
}

double BayesianHMM::calcTagPosteriorScore(MyTagIdType k, const int t, const std::vector<MyTagIdType> &tag_sent, MyWordIdType word_id)
{
    double word_prob = this->calcWordProbGivenTag(word_id, k);

    std::vector<MyTagIdType> ngram(n_);
    copy(tag_sent.begin() + t, tag_sent.begin() + t + 2, ngram.begin());
    double ngram_prob1 = this->calcTagNgramProb(ngram);

    double add_denominator = 0.0;
    double add_numerator = 0.0;
    if (tag_sent[t] == tag_sent[t + 1] && tag_sent[t] == tag_sent[t + 2])
    {
        add_denominator += 1.0;
        if (tag_sent[t] == tag_sent[t + 3])
        {
            add_numerator += 1.0;
        }
    }
    copy(tag_sent.begin() + t + 1, tag_sent.begin() + t + 3, ngram.begin());
    double ngram_prob2 = this->calcTagNgramProb(ngram, add_denominator, add_numerator);

    add_denominator = 0.0;
    add_numerator = 0.0;
    if (tag_sent[t + 1] == tag_sent[t + 2] && tag_sent[t + 1] == tag_sent[t + 3])
    {
        add_denominator += 1.0;
        if (tag_sent[t] == tag_sent[t + 4])
        {
            add_numerator += 1.0;
        }
    }
    if (tag_sent[t] == tag_sent[t + 2] && tag_sent[t + 1] == tag_sent[t + 3]) {
        add_denominator += 1.0;
        if (tag_sent[t] == tag_sent[t + 4]){
            add_numerator += 1.0;
        }
    }
    copy(tag_sent.begin() + t + 2, tag_sent.begin() + t + 4, ngram.begin());
    double ngram_prob3 = this->calcTagNgramProb(ngram, add_denominator, add_numerator);

    double score = word_prob * ngram_prob1 * ngram_prob2 * ngram_prob3;
    return score;
}
