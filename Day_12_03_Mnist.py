# Day_12_03_Mnist.py
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection


# 문제
# mnist dataset에 대해서 정확도를 계산하세요.

# 멀티 레이어 버전 90% 수준의 정확도를 달성하세요.

# mnist = tf.keras.datasets.mnist.load_data()
# print(type(mnist)) # tuple
# print(len(mnist))  # 2
# print(type(mnist[0]), type(mnist[1]))   #<class 'tuple'> <class 'tuple'>
# print(len(mnist[0]), len(mnist[1])) # 2 2

# tupel로 묶여 있음

# sparse버전만 존재

def softmax_mnist():

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28) 3차원/ 2차원을 취급
    print(y_train.shape, y_test.shape)  # (60000,) (10000,) 원핫이 아님

    x_train = x_train.reshape(-1, 784)
    x_test  = x_test.reshape(-1, 784)
    print(x_train[0])
    print(x_train.dtype)    #uint8

    # 문제
    # 표준화를 적용하세요.
    # scaler = preprocessing.StandardScaler()
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    # uint8 error

    x_train = np.float32(x_train)
    x_test = np.float32(x_test)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax))

    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                  loss = tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])   #

    model.fit(x_train, y_train, epochs = 10, verbose=2, batch_size=100)
    print('acc:', model.evaluate(x_test, y_test))

def multi_layers_mnist():

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28) 3차원/ 2차원을 취급
    print(y_train.shape, y_test.shape)  # (60000,) (10000,) 원핫이 아님

    x_train = x_train.reshape(-1, 784)
    x_test  = x_test.reshape(-1, 784)
    print(x_train[0])
    print(x_train.dtype)    #uint8

    # 문제
    # 표준화를 적용하세요.
    # scaler = preprocessing.StandardScaler()
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    # uint8 error

    x_train = np.float32(x_train)
    x_test = np.float32(x_test)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(256, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(128, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax))


    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                  loss = tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])   #

    model.fit(x_train, y_train, epochs = 10, verbose=2, batch_size=100)
    print('acc:', model.evaluate(x_test, y_test))


# softmax_mnist()
multi_layers_mnist()