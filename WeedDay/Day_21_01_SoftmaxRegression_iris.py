# Day_21_01_SoftmaxRegression_iris.py
import pandas as pd
from sklearn import preprocessing, model_selection
import numpy as np
import tensorflow as tf


def show_accuracy(preds, labels):
    preds_arg = np.argmax(preds, axis=1)
    y_arg = np.argmax(labels, axis=1)

    equals = (preds_arg == y_arg)
    print('acc :', np.mean(equals))


def show_accuracy_sparse(preds, labels):
    preds_arg = np.argmax(preds, axis=1)
    # y_arg = np.argmax(labels, axis=1)

    equals = (preds_arg == labels)
    print('acc :', np.mean(equals))


def softmax_iris():
    # 문제
    # 붓꽃 데이터 파일을 읽어오세요
    iris = pd.read_csv('data/iris(150).csv',
                       index_col=0)
    print(iris)

    # 문제
    # x, y를 만드세요 (y는 인코딩되어야 합니다)
    enc = preprocessing.LabelBinarizer()
    y = enc.fit_transform(iris.Species)
    print(y.shape)
    print(y[:3])

    # x = iris.values[:, :-1]
    # df = iris.drop(['Species'], axis=1)
    # x = df.values

    iris.drop(['Species'], axis=1, inplace=True)
    x = iris.values
    print(x.shape)
    print(x[:3])

    x = np.float32(x)

    data = model_selection.train_test_split(x, y, train_size=0.7)
    x_train, x_test, y_train, y_test = data

    # ----------------------------- #

    # 문제
    # 7대 3으로 나눠서 정확도를 구하세요
    n_features = x.shape[1]
    n_classes = y.shape[1]
    w = tf.Variable(tf.random_uniform([n_features, n_classes]))  # [4, 3]
    b = tf.Variable(tf.random_uniform([n_classes]))  # [3]

    ph_x = tf.placeholder(tf.float32)

    # (150, 3) = (150, 4) @ (4, 3)
    z = tf.matmul(ph_x, w) + b
    hx = tf.nn.softmax(z)

    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_train, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train, {ph_x: x_train})
        print(i, sess.run(loss, {ph_x: x_train}))

    # ---------------------------- #

    # 문제
    # show_accuracy 함수를 만들어서 결과를 표시하세요
    preds_train = sess.run(hx, {ph_x: x_train})
    preds_test = sess.run(hx, {ph_x: x_test})

    show_accuracy(preds_train, y_train)
    show_accuracy(preds_test, y_test)

    sess.close()


def softmax_iris_sparse():
    # 문제
    # 붓꽃 데이터 파일을 읽어오세요
    iris = pd.read_csv('data/iris(150).csv',
                       index_col=0)

    # 문제
    # x, y를 만드세요 (y는 인코딩되어야 합니다)
    enc = preprocessing.LabelEncoder()
    y = enc.fit_transform(iris.Species)
    print(y.shape)
    print(y[:3], y[-3:])

    # np.eye(3)[y]

    # x = iris.values[:, :-1]
    # df = iris.drop(['Species'], axis=1)
    # x = df.values

    iris.drop(['Species'], axis=1, inplace=True)
    x = iris.values
    print(x.shape)
    print(x[:3])

    x = np.float32(x)

    data = model_selection.train_test_split(x, y, train_size=0.7)
    x_train, x_test, y_train, y_test = data

    print(y_train.shape, y_test.shape)

    # ----------------------------- #

    # 문제
    # 7대 3으로 나눠서 정확도를 구하세요
    n_features = x.shape[1]
    n_classes = 3       # y.shape[1]
    w = tf.Variable(tf.random_uniform([n_features, n_classes]))  # [4, 3]
    b = tf.Variable(tf.random_uniform([n_classes]))  # [3]

    ph_x = tf.placeholder(tf.float32)

    # (150, 3) = (150, 4) @ (4, 3)
    z = tf.matmul(ph_x, w) + b
    hx = tf.nn.softmax(z)

    loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_train, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train, {ph_x: x_train})
        print(i, sess.run(loss, {ph_x: x_train}))

    # ---------------------------- #

    # 문제
    # show_accuracy 함수를 만들어서 결과를 표시하세요
    preds_train = sess.run(hx, {ph_x: x_train})
    preds_test = sess.run(hx, {ph_x: x_test})

    show_accuracy_sparse(preds_train, y_train)
    show_accuracy_sparse(preds_test, y_test)

    sess.close()


# softmax_iris()
# softmax_iris_sparse()
