#ifndef INCLUDE_GUARD_HOGE_HPP
#define INCLUDE_GUARD_HOGE_HPP

using MyWordIdType = unsigned int;
const MyWordIdType BEGIN_WORD_ID = 0;
const MyWordIdType END_WORD_ID = 0;
const MyWordIdType UNK_WORD_ID = 1;
const int SPECIAL_WORD_SIZE = 2;

using MyTagIdType = unsigned int;
const MyTagIdType BEGIN_TAG_ID = 0;
const MyTagIdType END_TAG_ID = 1;
const int SPECIAL_TAG_SIZE = 2;

const int NGRAM_SIZE = 3;

#endif // INCLUDE_GUARD_HOGE_HPP
