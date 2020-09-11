# Day_14_03_NovelBot.py
import tensorflow as tf
import numpy as np
from sklearn import preprocessing, model_selection


# 문제 1
# 니체의 저서를 읽고 니체풍으로 새로운 소설을 쓰세요
# BasicRnn_8과 Stock 코드를 병합해서 사용하도록 합니다
# (fit 함수에서 에러나지 않으면 성공)

# 문제 2
# 결과는 BasicRnn_8 코드와 동일하게 출력합니다
# (predict 함수 결과를 제대로 표시하면 성공)


def get_data(seq_length):
    f = open('data/nietzsche.txt', 'r', encoding='utf-8')
    long_text = f.read(10000).lower()
    f.close()

    # print(long_text)
    # print(len(long_text))

    lb = preprocessing.LabelBinarizer()
    onehot = lb.fit_transform(list(long_text))

    rng = [(i, i + seq_length) for i in range(len(long_text) - seq_length)]

    x = [onehot[s:e] for s, e in rng]
    y = [onehot[e] for s, e in rng]
    print(y[:3])

    x = np.float32(x)
    y = np.argmax(y, axis=1)
    print(x.shape, y.shape)     # (600833, 60, 57) (600833,)

    return x, y, lb.classes_


def show_sampling(sample_func):
    seq_length = 60
    x, y, vocab = get_data(seq_length)

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dense(len(vocab), activation=tf.keras.activations.softmax),
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])
    model.fit(x, y, epochs=5, verbose=2, batch_size=128)

    # 1번
    # if sample_func == None:
    #     old_fashion(model, x, vocab)

    # 2번
    # if sample_func:
    #     new_fashion(model, vocab, seq_length, y, sample_func)

    # 4번
    if sample_func.__name__ == 'temperature_pick':
        for temperature in [0.2, 0.5, 0.8, 1.0, 2.0]:
            print('-' * 30)
            new_fashion(model, vocab, seq_length, y, sample_func, temperature)


def old_fashion(model, x, vocab):
    preds = model.predict(x)
    print(preds.shape)

    preds_arg = np.argmax(preds, axis=1)
    print(preds_arg.shape)          # (940,)

    # print(long_text)
    # print('-' * 30)

    # for p in preds_arg:
    #     print(lb.classes_[p], end='')

    # print(lb.classes_[preds_arg])
    print(''.join(vocab[preds_arg]))


def new_fashion(model, vocab, seq_length, y, sample_func, temperature=0.0):
    start = np.random.randint(0, len(y)-1, 1)
    # print(start)
    start = start[0]

    indices = y[start:start+seq_length]
    # indices = list(indices)
    # print(indices)

    for i in range(100):
        xx = np.zeros([1, seq_length, len(vocab)])
        # print(xx.shape)         # (1, 60, 31)
        for j, pos in enumerate(indices):
            # print(j, pos)
            xx[0, j, pos] = 1

        preds = model.predict(xx)
        # print(preds)
        preds = preds[0]

        # t = np.argmax(preds)
        t = sample_func(preds) if temperature == 0.0 else sample_func(preds, temperature)
        print(vocab[t], end='')

        # list
        # indices.pop(0)
        # indices.append(t)

        # numpy
        indices[:-1] = indices[1:]
        indices[-1] = t
    print()


def weighted_pick(preds):
    t = np.cumsum(preds)        # 누적 합계
    return np.searchsorted(t, np.random.rand(1) * t[-1])[0]
    # print(t)
    # print(t[-1])
    # print(np.random.rand(1))
    # print(np.searchsorted(t, np.random.rand(1) * t[-1]))
    # [1 3 7 8 12 15]
    # 4 --> 2
    # 11 --> 4
    # [0.3 0.2 0.25 0.1 ...]
    # -> [0.3 0.5 0.75 0.85 ... 1.0]


def temperature_pick(preds, temperature):
    # preds = np.float64(preds)     # multinomial 함수에서만 사용
    preds = np.log(preds) / temperature
    preds = np.exp(preds)
    preds = preds / np.sum(preds)

    # preds가 10개짜리였다면, 반환값은 0이 10개 들어있는 배열
    # n의 갯수만큼 특정 위치를 1로 채웁니다
    # probas = np.random.multinomial(n=1, pvals=preds, size=1)
    # print(probas.shape)         # (1, 46)
    # return np.argmax(probas)
    return weighted_pick(preds)


# show_sampling(None)
# show_sampling(np.argmax)
# show_sampling(weighted_pick)
show_sampling(temperature_pick)

# WARNING:tensorflow
# NING:t -> k
# ING:tk -> m
# NG:tkm -> a
# G:tkma

