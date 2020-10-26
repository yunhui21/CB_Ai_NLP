# Da
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# def extrast(token_count, target, windows_size ):
#     start = max(target - windows_size, 0)
#     end = min(target + windows_size + 1, token_count)
#     return [tokens[i] for i in range(start, end) if i != target]
#     pass

def make_vocab_and_index(corpus, stop_words):
    def remove_stop_words(corpus, stop_words):
        # return [[i for i in corpus if i != stop_words]]
        return [[word for word in sent.split() if word not in stop_words] for sent in corpus]

    # 문제
    #
    def make_vocab(corpus_by_word):
        return sorted({w for words in corpus_by_word for w in words}) # 2차원을 1차원으로 풀어준다.


    corpus_by_word = remove_stop_words(corpus, stop_words)
    print(corpus_by_word)
    '''
    [['king', 'strong', 'man'], ['queen', 'wise', 'women'], 
    ['boy', 'young', 'man'], ['girl', 'young', 'women'], 
    ['prince', 'young', 'king'], ['princess', 'young', 'queen'], 
    ['man', 'strong'], ['women', 'pretty'], 
    ['prince', 'boy', 'king'], ['princess', 'girl', 'queen']]
    '''
    vocab = make_vocab(corpus_by_word)
    print(vocab)
    # ['boy', 'girl', 'king', 'man', 'pretty', 'prince', 'princess', 'queen', 'strong', 'wise', 'women', 'young']

    # 문제
    # corpus_by_word안의  단어를 vocab에 인덱스로 변환하세요.

    corpus_idx = [[vocab.index(w) for w in sent] for sent in corpus_by_word]
    print(corpus_idx)
    # [[2, 8, 3], [7, 9, 10], [0, 11, 3], [1, 11, 10], [5, 11, 2], [6, 11, 7], [3, 8], [10, 4], [5, 0, 2], [6, 1, 7]]

    return corpus_idx, vocab

def build_dataset(corpus_idx, n_classes, window_size, is_skipgram):
    xx, yy = [], []
    for sent in corpus_idx:
        for i, target in enumerate(sent):
            # print(i, target)
            stx = extract(len(sent), i, window_size, sent)

            if is_skipgram:
                for neighbor in stx:
                    xx.append(target)
                    yy.append(neighbor)
            else:
                xx.append(stx)
                yy.append(target)
    print(xx)
    print(yy)
# word2vec 에서 onehotvec을 하나의 소스로 작업한다. 전이학습


corpus = ['king is a strong man',
          'queen is a wise women',
          'boy is a young man',
          'girl is a young women',
          'prince is a young king',
          'princess is a young queen',
          'man is strong',
          'women is pretty',
          'prince is a boy will be king',
          'princess is a girl will be queen']

stop_words =['is', 'a', 'will', 'be']

corpus_idx, vocab = make_vocab_and_index(corpus, stop_words)

# build_dataset(corpus_idx, len(vocab), 1, is_skipgram=True)
build_dataset(corpus_idx, len(vocab), 1, is_skipgram=False)









