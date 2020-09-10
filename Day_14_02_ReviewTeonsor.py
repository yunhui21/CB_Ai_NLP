# Day_14_02_ReviewTeonsor.py

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing, model_selection


# 문제
# 'tensor'로 부터 x, y를 만들어서
# 케라스로 결과를 예측하세요.

def tensor_softmax():
    lb = preprocessing.LabelBinarizer()
    onehot = lb.fit_transform(list('tensor'))
    print(onehot)

    x = onehot[:-1]
    y = onehot[1:]

    x = np.float32(x)

    print(x.shape)   # (5, 6)
    print(y.shape)   # (1, 6)

    model = tf.keras.Sequential([
            tf.keras.layers.Dense(len(lb.classes_), activation=tf.keras.activations.softmax)
        ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.1),
                  loss=tf.keras.losses.categorical_crossentropy,
                  matrics=['acc'])
    model.fit(x, y, epochs=10)

    preds = model.predict(x)
    print(preds)

    preds_arg = np.argmax(preds, axis=1)
    print(preds_arg)
    print(lb.classes_[preds_arg])

def tensor_simple_rnn():
    lb = preprocessing.LabelBinarizer()
    onehot = lb.fit_transform(list('tensor'))
    print(onehot)

    x = onehot[:-1]
    y = onehot[1:]

    # x = np.float32([x])
    x = np.float32(x)

    # x = x.reshape(1, *x.shape)
    # y = y.reshape(1, *y.shape)

    x = x[np.newaxis]
    y = y[np.newaxis]

    # x = x[np.newaxis, :, :]
    # y = y[:, np.newaxis, :]
    print(x.shape, y.shape) # (1, 5, 6) (1, 5, 6)

    model = tf.keras.Sequential([
        tf.keras.layers.SimpleRNN(7, return_sequences=True),
        tf.keras.layers.Dense(len(lb.classes_), activation=tf.keras.activations.softmax)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.1),
                  loss=tf.keras.losses.categorical_crossentropy,
                  matrics=['acc'])
    model.fit(x, y, epochs=10, verbose=2)

    preds = model.predict(x)
    preds = preds[0]    #3차원을 2차원(추가된 코드)
    print(preds)

    preds_arg = np.argmax(preds, axis=1)
    print(preds_arg)
    print(lb.classes_[preds_arg])

def tensor_simple_rnn_sparse():
    lb = preprocessing.LabelBinarizer()
    onehot = lb.fit_transform(list('tensor'))
    print(onehot)

    x = onehot[:-1]
    y = onehot[1:]

    # x = np.float32([x])
    x = np.float32(x)
    y = np.argmax(y, axis=1)    #[0, 1, 4, 2, 3]
    y = y.reshape(-1, 1)        #로지스틱에서 사용하던 형태로 반환

    x = x[np.newaxis]
    y = y[np.newaxis]

    print(x.shape, y.shape) # (1, 5, 6) (1, 5)

    model = tf.keras.Sequential([
        tf.keras.layers.SimpleRNN(7, return_sequences=True),
        tf.keras.layers.Dense(len(lb.classes_), activation=tf.keras.activations.softmax)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.1),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  matrics=['acc'])
    model.fit(x, y, epochs=10, verbose=2)

    # sparse 버전에서 수정할것 없음.
    preds = model.predict(x)
    preds = preds[0]    #3차원을 2차원(추가된 코드)
    print(preds)

    preds_arg = np.argmax(preds, axis=1)
    print(preds_arg)
    print(lb.classes_[preds_arg])

# tensor_softmax()
# tensor_simple_rnn()
tensor_simple_rnn_sparse()