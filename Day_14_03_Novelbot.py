# Day_14_03_Novelbot.py

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing, model_selection

# 문제 1
# 니체의 저서를 읽고 니체풍으로 새로운 소설을 쓰세요.
# BasicRnn_8과 stock 코드를 병합해서 사용하도록 합니다.

# 문제 2
# 결과를
def get_data(seq_length):
    f = open('data/nietzsche.txt', 'r', encoding='utf-8')
    long_text = f.read().lower()
    f.close()

    print(long_text)

    lb = preprocessing.LabelBinarizer()
    onehot = lb.fit_transform(list(long_text))

    # seq_len = 60
    rng = [(i, i+seq_length) for i in range(len(long_text) - seq_length)]

    x = [onehot[s:e] for s, e in rng]
    y = [onehot[e] for s, e in rng]
    print(y[:3])

    x = np.float32(x)
    y = np.argmax(y, axis=1)
    print(x.shape, y.shape)

    return x, y, lb.classes_

def show_sampling(sample_func):
    seq_length = 60
    x, y, vocab = get_data(seq_length)

    model = tf.keras.Sequential([
            tf.keras.layers.LSTM(128),
            tf.keras.layers.Dense(len(vocab), activation=tf.keras.activations.softmax)
        ])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.1),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  matrics=['acc'])
    model.fit(x, y, epochs=5, verbose=2, batch_size=128)

    # 1번
    # if sample_func == None
    # old_fashion(model, x, vocab)

    # 2번
    # if sample_func:
    #     new_fashion(model, vocab, seq_length, y, sample_func)

    # 4번
    if sample_func. __name__ == 'temperature_pick':
        for temperature in [0.2, 0.5, 0.8, 1.0, 2.0]:
            print('-'*50)
            new_fashion(model, vocab, seq_length, y, sample_func, temperature)


def old_fashion(model, x, vocab):
    preds = model.predict(x)

    preds_arg = np.argmax(preds, axis=1)
    print(preds_arg.shape)

    # print(long_text)
    print(''.join(vocab[preds_arg]))


def new_fashion(model, vocab, seq_length, y, sample_func, temperatture=0.0):
    start = np.random.randint(0, len(y) - 1, 1)
    # print(start)
    start = start[0]

    indices = y[start:start + seq_length]
    # indices = list(indices)
    # print(indices)

    for i in range(100):
        xx = np.zeros([1, seq_length, len(vocab)])
        # print(xx.shape)     # (1,60,31)
        for j, pos in enumerate(indices):
            # print(j, pos)
            xx[0, j, pos] = 1
        preds = model.predict(xx)
        # print(k)
        preds = preds[0]

        # t = np.argmax(k)
        t = sample_func(preds)   if temperatture == 0.0 else sample_func(preds, temperatture)
        print(vocab[t], end='')

        # list
        # indices.pop(0)
        # indices.append(t)

        # numpy
        indices[:-1] = indices[1:]
        indices[-1] = t
    print()

def weighted_pick(preds):
    t = np.cumsum(preds)        # 누적 합계 softmax 전체합계 1이므로 마지막 값은 1일 나오야 한다.
    return np.searchsorted(t, np.random.rand(1)* t[-1])[0]

    # print(t)
    # print(t[-1])
    # print(np.random.rand(1))
    # print(np.searchsorted(t, np.random.rand(1)* t[-1]))     #미세한 오차로 인한 값이 들어올경우에 대비로 t[-1]
    # searchsoreted 정렬된 배열에서 어디로 들어갈지를 찾는것.
    # [ 1,3,5,2,8,11,12]
    # 4 -> 3
    # 11-> 5
    # [0,3, 0.2, 0.25, 0.1 ...]
    # ->[0.3, 0.5, 0.25, 0.05 ...]
    #
def temperature_pick(preds, temperature):
    # preds = np.float64(preds)     #multinormial함수에서만 사용
    preds = np.log(preds)/temperature
    preds = np.exp(preds)
    preds / np.sum(preds)

    # preds가 10개짜리 였다면, 반환값을 0이 10개 들어있는 배열
    # 0의 갯수만큼 특정 위치를 i로 채웁니다.
    # probas = np.random.multinomial(n=1, pvals=preds, size=1)
    # print(probas.shape)     # (1, 46)
    # return np.argmax(probas)
    return weighted_pick(preds)


# get_data()
# show_sampling(None) # old_fashion
# show_sampling(np.argmax) # new_fashion
# show_sampling(weighted_pick)
show_sampling(temperature_pick)
# WARNING:tensorf
# RNING:-> k
# NING:k-> m
# NG:km -> a
# G:kma (style을 반영하는 방법)