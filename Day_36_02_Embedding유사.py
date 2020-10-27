# # Da
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
#
# # def extrast(token_count, target, windows_size ):
# #     start = max(target - windows_size, 0)
# #     end = min(target + windows_size + 1, token_count)
# #     return [tokens[i] for i in range(start, end) if i != target]
# #     pass
#
# def make_vocab_and_index(corpus, stop_words):
#     def remove_stop_words(corpus, stop_words):
#         # return [[i for i in corpus if i != stop_words]]
#         return [[word for word in sent.split() if word not in stop_words] for sent in corpus]
#
#     # 문제
#     #
#     def make_vocab(corpus_by_word):
#         return sorted({w for words in corpus_by_word for w in words}) # 2차원을 1차원으로 풀어준다.
#
#
#     corpus_by_word = remove_stop_words(corpus, stop_words)
#     print(corpus_by_word)
#     '''
#     [['king', 'strong', 'man'], ['queen', 'wise', 'women'],
#     ['boy', 'young', 'man'], ['girl', 'young', 'women'],
#     ['prince', 'young', 'king'], ['princess', 'young', 'queen'],
#     ['man', 'strong'], ['women', 'pretty'],
#     ['prince', 'boy', 'king'], ['princess', 'girl', 'queen']]
#     '''
#     vocab = make_vocab(corpus_by_word)
#     print(vocab)
#     # ['boy', 'girl', 'king', 'man', 'pretty', 'prince', 'princess', 'queen', 'strong', 'wise', 'women', 'young']
#
#     # 문제
#     # corpus_by_word안의  단어를 vocab에 인덱스로 변환하세요.
#
#     corpus_idx = [[vocab.index(w) for w in sent] for sent in corpus_by_word]
#     print(corpus_idx)
#     # [[2, 8, 3], [7, 9, 10], [0, 11, 3], [1, 11, 10], [5, 11, 2], [6, 11, 7], [3, 8], [10, 4], [5, 0, 2], [6, 1, 7]]
#
#     return corpus_idx, vocab
#
# def build_dataset(corpus_idx, n_classes, window_size, is_skipgram):
#     xx, yy = [], []
#     for sent in corpus_idx:
#         for i, target in enumerate(sent):
#             # print(i, target)
#             stx = extract(len(sent), i, window_size, sent)
#
#             if is_skipgram:
#                 for neighbor in stx:
#                     xx.append(target)
#                     yy.append(neighbor)
#             else:
#                 xx.append(stx)
#                 yy.append(target)
#     print(xx)
#     print(yy)
# # word2vec 에서 onehotvec을 하나의 소스로 작업한다. 전이학습
#
#
# corpus = ['king is a strong man',
#           'queen is a wise women',
#           'boy is a young man',
#           'girl is a young women',
#           'prince is a young king',
#           'princess is a young queen',
#           'man is strong',
#           'women is pretty',
#           'prince is a boy will be king',
#           'princess is a girl will be queen']
#
# stop_words =['is', 'a', 'will', 'be']
#
# corpus_idx, vocab = make_vocab_and_index(corpus, stop_words)
#
# # build_dataset(corpus_idx, len(vocab), 1, is_skipgram=True)
# build_dataset(corpus_idx, len(vocab), 1, is_skipgram=False)
#
#
#
#
#
#
#
#
#
# Day_36_02_Word2Vec_Adv.py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gensim


# 문제
# callback을 사용해서 진행되는

def extract(token_count, target, window_size, tokens):
    start = max(target - window_size, 0)
    end = min(target + window_size + 1, token_count)
    return [tokens[i] for i in range(start, end) if i != target]


def make_vocab_and_index(corpus, stop_words):
    # 문제
    # 불용어를 제거해서 2차원 단어 목록을 만드세요
    def remove_stop_words(corpus, stop_words):
        return [[word for word in sent.split() if word not in stop_words] for sent in corpus]

    # 문제
    # 중복되지 않는 단어로 구성된 1차원 리스트를 만드세요
    def make_vocab(corpus_by_word):
        return sorted({w for words in corpus_by_word for w in words})

    corpus_by_word = remove_stop_words(corpus, stop_words)
    print(corpus_by_word)
    # [['king', 'strong', 'man'], ['queen', 'wise', 'woman'],
    # ['boy', 'young', 'man'], ['girl', 'young', 'woman'],
    # ['prince', 'young', 'king'], ['princess', 'young', 'queen'],
    # ['man', 'strong'], ['woman', 'pretty'],
    # ['prince', 'boy', 'king'], ['princess', 'girl', 'queen']]

    vocab = make_vocab(corpus_by_word)
    print(vocab)
    # ['boy', 'girl', 'king', 'man', 'pretty', 'prince', 'princess',
    # 'queen', 'strong', 'wise', 'woman', 'young']

    # 문제
    # corpus_by_word에 포함된 단어를 단어장의 인덱스로 변환하세요
    corpus_idx = [[vocab.index(w) for w in sent] for sent in corpus_by_word]
    print(corpus_idx)
    # [[2, 8, 3], [7, 9, 10], [0, 11, 3], [1, 11, 10], [5, 11, 2],
    # [6, 11, 7], [3, 8], [10, 4], [5, 0, 2], [6, 1, 7]]

    return corpus_idx, vocab


def build_dataset(corpus_idx, n_classes, window_size, is_skipgram):
    xx, yy = [], []
    for sent in corpus_idx:
        for i, target in enumerate(sent):
            # print(i, target)

            ctx = extract(len(sent), i, window_size, sent)

            if is_skipgram:
                for neighbor in ctx:
                    xx.append(target)
                    yy.append(neighbor)
            else:
                xx.append(ctx)
                yy.append(target)

    print(xx)
    print(yy)

    return make_onehot(xx, yy, n_classes, is_skipgram)

def make_onehot(xx, yy, n_classes, is_skipgram):
    x = np.zeros([len(xx), n_classes], dtype=np.float32)
    y = np.zeros([len(xx), n_classes], dtype=np.float32)

    for i, (input, label) in enumerate(zip(xx,yy)):
        print(i, input, label)
        y[i, label] = 1 # y번째 데이터는

        if is_skipgram:
            x[i, input] = 1  # y번째 데이터는
        else:
            z = [[int(pos == j) for j in range(n_classes)] for pos in input]
            print(z)
            #23 [5, 2] 0
            # [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            #  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
            x[i] = np.mean(z, axis=0)
            # [[0, 0, 0.5, 0, 0, .5, 0, 0, 0, 0, 0, 0],  평균값
            print(y)
            # [[0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
            #  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
            # print(x)
            # skipgram
            # [[0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
            #  [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
    return x, y

def show_word2vec(corpus_idx, vocab, window_size, is_skipgram):
    n_classes = len(vocab)
    x, y = build_dataset(corpus_idx, n_classes, window_size, is_skipgram)

    n_embeds = 2 # 단어 하나의 표현을 2개의 숫자로 나타낸다. 시각화를 편하게 하기 위해서..2개만 가져본다.
    n_features = x.shape[1]

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input([n_classes]))
    model.add(tf.keras.layers.Dense(n_embeds))
    model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))

    # model.compile(optimizer='sgd', loss=tf.keras.losses.categorical_crossentropy)
    model.compile(optimizer='adam', loss=tf.keras.losses.categorical_crossentropy)

    model.fit(x, y, epochs=20000, verbose =0, callbacks=EveryEpoch())



    dense = model.get_layer(index =0)
    vectors = tf.keras.backend.get_value(dense.weights[0])

    print(vectors)
    print(vectors.shape)
    print(vocab)
    #['boy', 'girl', 'king', 'man', 'pretty', 'prince', 'princess', 'queen',
    # 'strong', 'wise', 'woman', 'young']

    for i in range(len(vocab)):
        print(vocab[i], vectors[i])
        '''
        boy [-0.55311304 -0.6598454 ]
        girl [-0.43132216  0.27653542]
        king [ 0.5622876  -0.43781006]
        man [ 0.35233974 -0.16005568]
        pretty [0.27614    0.07164413]
        prince [-0.38544694  0.1587612 ]
        princess [ 0.30820334 -0.21224508]
        queen [ 0.25214833 -0.02614678]
        strong [-0.4299036   0.27791312]
        wise [-0.08832226 -0.50934196]
        woman [-0.09930269  0.3137353 ]
        young [ 0.4697932 -0.535046 ]
        '''
    show_similarity(vectors, vocab, 'skip-grm' if is_skipgram else 'cbow')

def show_similarity(vectors, vocab, title):
    for word, (x1, x2) in zip(vocab, vectors):
        plt.text(x1, x2, word)

    ax_min = np.min(vectors, axis=0) - 1
    ax_max = np.max(vectors, axis=0) + 1

    plt.xlim(ax_min[0], ax_max[0])
    plt.ylim(ax_min[1], ax_max[1])

    plt.title(title)
    plt.show()


class EveryEpoch(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 100 == 0:
            print(epoch, ':', logs['loss'])

def show_word2vec_by_gensim(corpus, vocab, stop_words, is_skipgram):
    vectors = gensim.models

    def remove_stop_words(corpus, stop_words):
        return [[word for word in sent.split() if word not in stop_words] for sent in corpus]

    corpus_by_word = remove_stop_words(corpus, stop_words)
    vectors = gensim.models.Word2Vec(corpus_by_word,
                                            size= 2, window= 1, min_count=1,
                                            sg = is_skipgram)
    print(vectors)
    print(vectors.wv)
    print(vectors.wv.vectors)

    show_similarity(vectors.wv.vectors, vocab, 'skip-gram' if is_skipgram else 'cbow')



corpus = ['king is a strong man',
          'queen is a wise woman',
          'boy is a young man',
          'girl is a young woman',
          'prince is a young king',
          'princess is a young queen',
          'man is strong',
          'woman is pretty',
          'prince is a boy will be king',
          'princess is a girl will be queen']
stop_words = ['is', 'a', 'will', 'be']

corpus_idx, vocab = make_vocab_and_index(corpus, stop_words)

# show_word2vec(corpus_idx, vocab, 1, is_skipgram=True)
# show_word2vec(corpus_idx, vocab, 1, is_skipgram=False)
# show_word2vec_by_gensim(corpus, vocab, stop_words, is_skipgram=True )
show_word2vec_by_gensim(corpus, vocab, stop_words, is_skipgram=False )






