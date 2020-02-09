
#include "bayesian_hmm.hpp"

#include "ProgressBar.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>

BayesianHMM::BayesianHMM(const int tag_size, const MyWordIdType vocab_size, const double alpha, const double beta) : tag_size_(tag_size + SPECIAL_TAG_SIZE), vocab_size_(vocab_size + SPECIAL_TAG_SIZE), alpha_(alpha), beta_(beta)
{
    std::random_device seed_gen;
    random_generator_.seed(seed_gen());
}

std::string BayesianHMM::join(const std::vector<MyWordIdType> &v, const std::string delim) const
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
void BayesianHMM::addNgramParameter(const std::vector<MyTagIdType> &ngram, const int recursive)
{
    assert(static_cast<int>(ngram.size()) == n_);
    assert(static_cast<int>(ngram.size()) >= recursive);
    assert(recursive >= 1);

    for (int i = 0; i < n_; i++)
    {
        std::vector<MyTagIdType> tmp(n_ - i);
        copy(ngram.begin() + i, ngram.begin() + n_, tmp.begin());
        std::string ngram_string = this->join(tmp, DELIMITER);
        tag_ngram_count_.emplace(ngram_string, 0);
        tag_ngram_count_[ngram_string] += 1;
        if (i >= recursive - 1)
        {
            break;
        }
    }
}

// ここはスムージングしているのか不明
void BayesianHMM::removeNgramParameter(const std::vector<MyTagIdType> &ngram, const int recursive)
{
    assert(static_cast<int>(ngram.size()) == n_);
    assert(static_cast<int>(ngram.size()) >= recursive);
    assert(recursive >= 1);

    for (int i = 0; i < n_; i++)
    {
        std::vector<MyTagIdType> tmp(n_ - i);
        copy(ngram.begin() + i, ngram.begin() + n_, tmp.begin());
        std::string ngram_string = this->join(tmp, DELIMITER);
        tag_ngram_count_[ngram_string] -= 1;
        assert(tag_ngram_count_[ngram_string] >= 0);
        if (tag_ngram_count_[ngram_string] == 0)
        {
            tag_ngram_count_.erase(ngram_string);
        }
        if (i >= recursive - 1)
        {
            break;
        }
    }
}

void BayesianHMM::addWordEmissionParameter(const MyWordIdType word_id, const MyTagIdType k)
{
    std::string word_emission_string = std::to_string(k) + DELIMITER + std::to_string(word_id);
    word_emission_count_.emplace(word_emission_string, 0);
    word_emission_count_[word_emission_string] += 1;
}

void BayesianHMM::removeWordEmissionParameter(const MyWordIdType word_id, const MyTagIdType k)
{
    std::string word_emission_string = std::to_string(k) + DELIMITER + std::to_string(word_id);
    word_emission_count_[word_emission_string] -= 1;
    assert(word_emission_count_[word_emission_string] >= 0);
    if (word_emission_count_[word_emission_string] == 0)
    {
        word_emission_count_.erase(word_emission_string);
    }
}

MyTagIdType BayesianHMM::samplingTthTag(const int t, const std::vector<MyTagIdType> &tag_sent, const MyWordIdType word_id)
{
    std::vector<double> scores(tag_size_, 0.0);
    double sum = 0.0;
    for (int k = SPECIAL_TAG_SIZE; k < tag_size_; k++)
    {
        double score = this->calcTagPosteriorScore(static_cast<MyTagIdType>(k), t, tag_sent, word_id);
        scores[k] = score;
        sum += score;
    }
    std::uniform_real_distribution<double> dist(0.0, sum);
    double r = dist(random_generator_);
    sum = 0.0;
    int sampled_k = -1;
    for (int k = SPECIAL_TAG_SIZE; k < tag_size_; k++)
    {
        sum += scores[k];
        if (sum >= r)
        {
            sampled_k = k;
            break;
        }
    }
    assert(sampled_k >= SPECIAL_TAG_SIZE);
    assert(sampled_k < tag_size_);

    return static_cast<MyTagIdType>(sampled_k);
}

// 実際のt+2番目のタグをサンプリング(BEGINの分を2つ文頭に足している,trigramの場合)
void BayesianHMM::gibbsSamplingTthTag(const int t, std::vector<MyTagIdType> &tag_sent, const std::vector<MyWordIdType> &word_sent)
{
    assert(t >= 0);
    assert(t + 2 < static_cast<int>(tag_sent.size()));
    assert(word_sent.size() == tag_sent.size());
    // t-th tag に関連するパラメータの削除
    // (tag_{t-2}, tag_{t-1}, tag_{t}), (tag_{t-1}, tag_{t}, tag_{t+1}), (tag_{t}, tag_{t+1}, tag_{t+2})
    for (int i = 0; i < n_; i++)
    {
        std::vector<MyTagIdType> ngram(n_);
        copy(tag_sent.begin() + t + i, tag_sent.begin() + t + i + n_, ngram.begin());
        this->removeNgramParameter(ngram, (n_ - i));
    }

    this->removeWordEmissionParameter(word_sent[t + 2], tag_sent[t + 2]);

    MyTagIdType sampled_k = this->samplingTthTag(t, tag_sent, word_sent[t + 2]);
    tag_sent[t + 2] = sampled_k;
    this->addWordEmissionParameter(word_sent[t + 2], tag_sent[t + 2]);

    // t-th tag に関連するパラメータの追加
    // (tag_{t-2}, tag_{t-1}, tag_{t}), (tag_{t-1}, tag_{t}, tag_{t+1}), (tag_{t}, tag_{t+1}, tag_{t+2})
    for (int i = 0; i < n_; i++)
    {
        std::vector<MyTagIdType> ngram(n_);
        copy(tag_sent.begin() + t + i, tag_sent.begin() + t + i + n_, ngram.begin());
        this->addNgramParameter(ngram, (n_ - i));
    }
}

double BayesianHMM::calcWordProbGivenTag(const MyWordIdType word_id, const MyTagIdType k) const
{
    double k_count = 0.0;
    auto itr = tag_ngram_count_.find(std::to_string(k));
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

double BayesianHMM::calcTagNgramProb(const std::vector<MyTagIdType> &ngram, const double add_denominator, const double add_numerator) const
{
    assert(static_cast<int>(ngram.size()) == n_);
    assert(add_denominator >= 0.0);
    assert(add_numerator >= 0.0);
    assert(add_denominator >= add_numerator);

    double ngram_count = 0.0;
    std::string ngram_string = this->join(ngram, DELIMITER);
    auto itr = tag_ngram_count_.find(ngram_string);
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

double BayesianHMM::calcTagPosteriorScore(const MyTagIdType k, const int t, const std::vector<MyTagIdType> &tag_sent, const MyWordIdType word_id) const
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
    if (tag_sent[t] == tag_sent[t + 2] && tag_sent[t + 1] == tag_sent[t + 3])
    {
        add_denominator += 1.0;
        if (tag_sent[t] == tag_sent[t + 4])
        {
            add_numerator += 1.0;
        }
    }
    copy(tag_sent.begin() + t + 2, tag_sent.begin() + t + 4, ngram.begin());
    double ngram_prob3 = this->calcTagNgramProb(ngram, add_denominator, add_numerator);

    double score = word_prob * ngram_prob1 * ngram_prob2 * ngram_prob3;
    return score;
}

void BayesianHMM::Train(std::vector<std::vector<MyWordIdType>> &corpus, std::vector<std::vector<MyTagIdType>> &tag_corpus, const int epoch)
{
    assert(corpus.size() == tag_corpus.size());

    this->initialize(corpus, tag_corpus);
    for (int e = 0; e < epoch; e++)
    {
        const unsigned int bar_width = 40;
        ProgressBar progress_bar(corpus.size(), bar_width);

        std::vector<int> rand_vec(corpus.size());
        std::iota(rand_vec.begin(), rand_vec.end(), 0);
        std::shuffle(rand_vec.begin(), rand_vec.end(), random_generator_);
        for (int i = 0; i < static_cast<int>(corpus.size()); i++)
        {
            ++progress_bar;
            progress_bar.display();

            int r = rand_vec[i];
            for (int t = 0; t < static_cast<int>(corpus[r].size()) - 2 * (n_ - 1); t++)
            {
                this->gibbsSamplingTthTag(t, tag_corpus[r], corpus[r]);
            }
        }
        progress_bar.done();

        double score = this->CalcTagScoreGivenCorpus(corpus, tag_corpus);
        std::cout << "score = " << score << " (lower is better)" << std::endl;
    }
}

//
// スコアをエントロピーにすると下がっているようには見えなかった。
// エントロピーの計算 p * log(p)
// lower is better
double BayesianHMM::CalcTagScoreGivenCorpus(const std::vector<std::vector<MyWordIdType>> &corpus, const std::vector<std::vector<MyTagIdType>> &tag_corpus) const
{
    double score = 0.0;
    double word_count = 0.0;
    for (int i = 0; i < static_cast<int>(corpus.size()); i++)
    {
        for (int t = 0; t < static_cast<int>(corpus[i].size()) - 2 * (n_ - 1); t++)
        {
            std::vector<double> scores(tag_size_, 0.0);
            double sum = 0.0;
            for (int k = SPECIAL_TAG_SIZE; k < tag_size_; k++)
            {
                MyWordIdType word_id = corpus[i][t + (n_ - 1)];
                double score = this->calcTagPosteriorScore(static_cast<MyTagIdType>(k), t, tag_corpus[i], word_id);
                scores[k] = score;
                sum += score;
            }
            MyTagIdType k = static_cast<MyTagIdType>(tag_corpus[i][t + (n_ - 1)]);
            double prob = scores[k] / sum;
            // entropy += prob * std::log2(prob);
            score += std::log2(prob);
            word_count += 1.0;
        }
    }

    score = -1.0 * score / word_count;
    return score;
}

void BayesianHMM::initialize(const std::vector<std::vector<MyWordIdType>> &corpus, std::vector<std::vector<MyTagIdType>> &tag_corpus)
{
    std::uniform_int_distribution<int> dist(SPECIAL_TAG_SIZE, tag_size_);
    for (int i = 0; i < static_cast<int>(tag_corpus.size()); i++)
    {
        for (int t = 0; t < n_ - 1; t++)
        {
            assert(tag_corpus[i][t] == BEGIN_TAG_ID);
            assert(corpus[i][t] == BEGIN_WORD_ID);
        }
        for (int t = n_ - 1; t < static_cast<int>(tag_corpus[i].size()) - (n_ - 1); t++)
        {
            MyTagIdType random_k = static_cast<MyTagIdType>(dist(random_generator_));
            tag_corpus[i][t] = random_k;

            std::vector<MyTagIdType> ngram(n_);
            copy(tag_corpus[i].begin() + t - (n_ - 1), tag_corpus[i].begin() + t + 1, ngram.begin());
            this->addNgramParameter(ngram, n_);
            this->addWordEmissionParameter(corpus[i][t], random_k);
        }
        for (int t = static_cast<int>(tag_corpus[i].size()) - (n_ - 1); t < static_cast<int>(tag_corpus[i].size()); t++)
        {
            assert(tag_corpus[i][t] == END_TAG_ID);
            assert(corpus[i][t] == END_WORD_ID);
            std::vector<MyTagIdType> ngram(n_);
            copy(tag_corpus[i].begin() + t - (n_ - 1), tag_corpus[i].begin() + t + 1, ngram.begin());
            this->addNgramParameter(ngram, n_);
        }
    }
}