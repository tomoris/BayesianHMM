#!/usr/bin/env python
# -*- coding:utf-8 -*-

import py_bhmm
import argparse


BEGIN_TAG_ID = 0
END_TAG_ID = 1
BEGIN_WORD_ID = 0
END_WORD_ID = 0
UNK_WORD_ID = 1
vocab_size = 2

n = 3

def load_corpus(file_name):
    global vocab_size
    w2i = {}
    corpus = []
    tag_corpus = []
    for line in open(file_name):
        print(line)
        tokens = line.rstrip().split()
        corpus.append([BEGIN_WORD_ID for i in range(n - 1)])
        tag_corpus.append([BEGIN_TAG_ID for i in range(n - 1)])
        for token in tokens:
            if token not in w2i:
                w2i[token] = vocab_size
                vocab_size += 1
            corpus[-1].append(w2i[token])
            tag_corpus[-1].append(2)
        corpus[-1] += [END_WORD_ID for i in range(n - 1)]
        tag_corpus[-1] += [END_TAG_ID for i in range(n - 1)]
    return corpus, tag_corpus, w2i

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--file', help='training file', type=str, required=True)
    parser.add_argument('--pos', help='pos size', type=int, default=10)
    parser.add_argument('--epoch', help='epoch', type=int, default=100)
    parser.add_argument('--alpha', help='hyperparameter (alpha > 0, it is better that alpha is lower than 1.0)', type=float, default=0.1)
    parser.add_argument('--beta', help='hyperparameter (beta > 0, it is better that beta is lower than 1.0)', type=float, default=0.1)
    args = parser.parse_args()

    corpus, tag_corpus, w2i = load_corpus(args.file)
    hmm = py_bhmm.BayesianHMM(args.pos, data_container.GetWordVocabSize(), args.alpha, args.beta)
    tag_corpus = hmm.Train(corpus, tag_corpus, args.epoch)

    # or you can rewrite this
    # data_container = py_bhmm.DataContainer(args.file, " ")
    # hmm = py_bhmm.BayesianHMM(args.pos, data_container.GetWordVocabSize(), args.alpha, args.beta)
    # data_container.tag_corpus = hmm.Train(data_container.corpus, data_container.tag_corpus, args.epoch)

    return

if __name__ == "__main__":
    main()