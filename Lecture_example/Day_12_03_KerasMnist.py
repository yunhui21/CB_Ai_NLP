# Day_12_03_KerasMnist.py
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection

# 문제
# mnist 데이터셋에 대해 정확도를 계산하세요
# (softmax 알고리즘, 90% 이상 목표)

# 문제
# 멀티 레이어 버전을 구축해서 98% 수준의 정확도를 달성하세요

# mnist = tf.keras.datasets.mnist.load_data()
# print(type(mnist))
# print(len(mnist))
# print(type(mnist[0]), type(mnist[1]))
# print(len(mnist[0]), len(mnist[1]))


# sparse 버전만 존재
def softmax_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28)
    print(y_train.shape, y_test.shape)  # (60000,) (10000,)

    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)
    print(x_train[0])
    print(x_train.dtype)

    # 문제
    # 표준화를 적용하세요
    # scaler = preprocessing.StandardScaler()
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    x_train = np.float32(x_train)
    x_test = np.float32(x_test)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax))

    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])

    model.fit(x_train, y_train, epochs=10, verbose=2, batch_size=100)
    print('acc :', model.evaluate(x_test, y_test))


def multi_layers_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)

    scaler = preprocessing.MinMaxScaler()
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    x_train = np.float32(x_train)
    x_test = np.float32(x_test)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(256, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(128, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax))

    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])

    # model.fit(x_train, y_train, epochs=10, verbose=2, batch_size=100)
    model.fit(x_train, y_train, epochs=10, verbose=2, batch_size=100,
              validation_split=0.2)
    print('acc :', model.evaluate(x_test, y_test))


# softmax_mnist()
multi_layers_mnist()




