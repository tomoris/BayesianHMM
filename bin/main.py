#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

import matplotlib
matplotlib.use('Agg') # 追加
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import py_bhmm


BEGIN_TAG_ID = 0
END_TAG_ID = 1
BEGIN_WORD_ID = 0
END_WORD_ID = 0
UNK_WORD_ID = 1
w2i = {"<unk>": UNK_WORD_ID}
pos2i = {}
vocab_size = 2

n = 3


def load_corpus(file_name, delimiter = " ", pos_delimiter = None, train=True):
    global vocab_size
    global w2i
    global pos2i
    corpus = []
    tag_corpus = []
    pos_corpus = []
    for line in open(file_name):
        tokens = line.rstrip().split(delimiter)
        corpus.append([BEGIN_WORD_ID for i in range(n - 1)])
        tag_corpus.append([BEGIN_TAG_ID for i in range(n - 1)])
        pos_corpus.append([-1 for i in range(n - 1)])
        for token in tokens:
            if pos_delimiter is not None:
                token_sp = token.split(pos_delimiter)
                word = token_sp[0]
                pos = token_sp[1]
            else:
                word = token
                pos = None
            if word not in w2i:
                if train:
                    w2i[word] = vocab_size
                    vocab_size += 1
            if pos not in pos2i:
                if train:
                    pos2i[pos] = len(pos2i)
            corpus[-1].append(w2i.get(word, UNK_WORD_ID))
            tag_corpus[-1].append(2)
            pos_corpus[-1].append(pos2i[pos])
        corpus[-1] += [END_WORD_ID for i in range(n - 1)]
        tag_corpus[-1] += [END_TAG_ID for i in range(n - 1)]
        pos_corpus[-1] += [-1 for i in range(n - 1)]
    return corpus, tag_corpus, pos_corpus, w2i, pos2i


def show_frequent_co_occurrence(corpus, tag_corpus, pos_corpus, w2i, pos2i, tag_size):
    i2w = {v: k for k, v in w2i.items()}
    tag2word_count = {}
    for (sent, tag_sent) in zip(corpus, tag_corpus):
        for t in range(len(sent)):
            if t < n or t > len(sent) - n:
                continue
            token = sent[t]
            tag = tag_sent[t]
            if tag not in tag2word_count:
                tag2word_count[tag] = {}
            if token not in tag2word_count[tag]:
                tag2word_count[tag][token] = 0
            tag2word_count[tag][token] += 1

    for tag, word_count_dict in tag2word_count.items():
        i = 0
        for word_id, count in sorted(word_count_dict.items(), key=lambda x: x[1], reverse=True):
            if i > 10:
                break
            i += 1
            word = i2w[word_id]
            print("tag {0}, {1} ({2})".format(tag, word, count))

    if len(pos2i) != 1:
        co_occurrence_mat = np.zeros((tag_size, len(pos2i)), dtype=float)
        for (tag_sent, pos_sent) in zip(tag_corpus, pos_corpus):
            for t in range(len(tag_sent)):
                if t < n or t > len(tag_sent) - n:
                    continue
                tag_id = tag_sent[t] - 2 # special tag の分
                pos_id = pos_sent[t]
                co_occurrence_mat[tag_id][pos_id] += 1
        co_occurrence_mat = (co_occurrence_mat.T / co_occurrence_mat.sum(axis=1)).T
        pos_list = [pos for pos, i in sorted(pos2i.items(), key=lambda x:x[1])]
        plt.figure()
        sns.heatmap(co_occurrence_mat, xticklabels=pos_list)
        plt.savefig('heatmap.png')
        plt.close()

    return


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--file', help='training file', type=str, required=True)
    parser.add_argument('--testfile', help='test file', type=str, default=None)
    parser.add_argument('--delimiter', help='token delimiter', type=str, default=" ")
    parser.add_argument('--posdelimiter', help='pos delimiter', type=str, default=None)
    parser.add_argument('--tag', help='tag size', type=int, default=10)
    parser.add_argument('--epoch', help='epoch', type=int, default=100)
    parser.add_argument('--alpha', help='hyperparameter (alpha > 0, it is better that alpha is lower than 1.0)', type=float, default=0.1)
    parser.add_argument('--beta', help='hyperparameter (beta > 0, it is better that beta is lower than 1.0)', type=float, default=0.1)
    parser.add_argument('--threads', help='number of maximum threads', type=int, default=1)
    args = parser.parse_args()

    corpus, tag_corpus, pos_corpus, w2i, pos2i = load_corpus(args.file, args.delimiter, args.posdelimiter)
    if args.testfile is not None:
        test_corpus, test_tag_corpus, test_pos_corpus, w2i, pos2i = load_corpus(args.testfile, args.delimiter, args.posdelimiter)
    global vocab_size
    hmm = py_bhmm.BayesianHMM(args.tag, vocab_size, args.alpha, args.beta)
    tag_corpus = hmm.Train(corpus, tag_corpus, args.epoch, args.threads)

    if args.testfile is None:
        tag_corpus = hmm.Test(corpus, tag_corpus, args.threads)
        show_frequent_co_occurrence(corpus, tag_corpus, pos_corpus, w2i, pos2i, args.tag)
    else:
        test_tag_corpus = hmm.Test(test_corpus, test_tag_corpus, args.threads)
        show_frequent_co_occurrence(test_corpus, test_tag_corpus, test_pos_corpus, w2i, pos2i, args.tag)

    # or you can rewrite this
    # data_container = py_bhmm.DataContainer(args.file, " ")
    # hmm = py_bhmm.BayesianHMM(args.pos, data_container.GetWordVocabSize(), args.alpha, args.beta)
    # data_container.tag_corpus = hmm.Train(data_container.corpus, data_container.tag_corpus, args.epoch, args.threads)

    return


if __name__ == "__main__":
    main()