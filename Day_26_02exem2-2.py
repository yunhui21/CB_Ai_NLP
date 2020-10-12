# Day_26_02exem2-2.py

# 문제
# mnist test 데이터에 대해 sgd  정확도를 구현하세요.
# mnist의 shape을 변경해서는 안됩니다.

import tensorflow as tf
from sklearn import preprocessing
import numpy as np



def softmax_mnist():

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28) 3차원/ 2차원을 취급
    print(y_train.shape, y_test.shape)  # (60000,) (10000,) 원핫이 아님

    x_train = x_train.reshape(-1, 784)
    x_test  = x_test.reshape(-1, 784)
    # print(x_train[0])
    # print(x_train.dtype)    #uint8

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

    model.fit(x_train, y_train, epochs = 1000, verbose=2, batch_size=100)
    print('acc:', model.evaluate(x_test, y_test))

softmax_mnist()