# Day_12_02_KerasBasic.py
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection

# 문제
# 소프트맥스 리그레션을 케라스로 구현하세요.

# Dense버전을 saprse 버전으로 수정하세요.

def softmax_keras_dense():
    x = [[1, 2],    # c
         [2, 1],
         [4, 5],    # b
         [5, 4],
         [8, 9],    # c
         [9, 8]]

    y = [[0, 0, 1],
         [0, 0, 1],
         [1, 0, 0],
         [1, 0, 0],
         [1, 0, 0],
         [1, 0, 0]]

    x = np.float32(x)
    y = np.float32(y)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(3, activation=tf.keras.activations.softmax))

    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                  loss = tf.keras.losses.categorical_crossentropy,
                  metrics=['acc'])   #

    model.fit(x, y, epochs = 100, verbose=2)

    print('acc:', model.evaluate(x, y))

    preds = model.predict(x)
    print(preds)


def softmax_keras_sparse():

    x = [[1, 2],    # c
         [2, 1],
         [4, 5],    # b
         [5, 4],
         [8, 9],    # c
         [9, 8]]

    y = [2,2,1,1,0,0]

    x = np.float32(x)
    y = np.float32(y)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(3, activation=tf.keras.activations.softmax))

    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                  loss = tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])   #

    model.fit(x, y, epochs = 100, verbose=2)
    print('acc:', model.evaluate(x, y))

    # preds = model.predict(x)
    # print(preds)

# softmax_keras_dense()
softmax_keras_sparse()