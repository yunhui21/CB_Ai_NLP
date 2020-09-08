# Day_12_02_KerasBasic.py
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection

# 문제
# 소프트맥스 리그레션을 케라스로 구현하세요.

# 문제
# Dense버전을 saprse 버전으로 수정하세요.

# 문제
# iris 데이터에 대해
# 70%로 학습하고 30%에 대해 정확도를 예측하세요.


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


def softmax_iris_dense():
    iris = pd.read_csv('data/iris(150).csv', index_col=0)
    # print(iris) #[150 rows x 5 columns]

    x = np.float32(iris.values[:, :-1])
    y = preprocessing.LabelBinarizer().fit_transform(iris.values[:,  -1])
    # print(x.shape, y.shape) #(150, 4) (150, 3)
    # print(y[:3]) #[[1 0 0] [1 0 0] [1 0 0]]
    # print(x.dtype)  #object
    # print(x[:3])    #[[5.1 3.5 1.4 0.2] [4.9 3.0 1.4 0.2] [4.7 3.2 1.3 0.2]]

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.7)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(3, activation=tf.keras.activations.softmax))

    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                  loss = tf.keras.losses.categorical_crossentropy,
                  metrics=['acc'])   #

    model.fit(x_train, y_train, epochs = 100, verbose=2)

    print('acc:', model.evaluate(x_test, y_test))

    # preds = model.predict(x)
    # print(preds)


def softmax_iris_sparse():

    iris = pd.read_csv('data/iris(150).csv', index_col=0)
    # print(iris) #[150 rows x 5 columns]

    y = preprocessing.LabelBinarizer().fit_transform(iris.Species)

    iris.drop(['Species'], axis=1, inplace=True)

    x = iris.values
    print(x.shape, y.shape) #(150, 4) (150, 3)
    print(y[:3]) #[[1 0 0] [1 0 0] [1 0 0]]
    print(x.dtype)  #object
    print(x[:3])    #[[5.1 3.5 1.4 0.2] [4.9 3.0 1.4 0.2] [4.7 3.2 1.3 0.2]]

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.7)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(3, activation=tf.keras.activations.softmax))

    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                  loss = tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])   #

    model.fit(x_train, y_train, epochs = 10, verbose=2)

    print('acc:', model.evaluate(x_test, y_test))

    # preds = model.predict(x)
    # print(preds)


# softmax_keras_dense()
# softmax_keras_sparse()
# softmax_iris_dense()
softmax_iris_sparse()