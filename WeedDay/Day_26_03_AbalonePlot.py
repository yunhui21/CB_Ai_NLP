# Day_26_03_AbalonePlot.py
# Day_26_02_Abalone.py
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import model_selection, preprocessing
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=1000)

# 문제 1
# 26-2 파일을 멀티플 리그레션으로 변환하세요

# 문제 2
# 10 에포크마다 발생하는 손실과 정확도를 그래프로 그리세요 (1,000 에포크)

def get_data():
    names = ['Sex', 'Length', 'Diamete', 'Height',
             'Whole', 'Shucked', 'Viscera', 'Shell', 'Rings']

    abalone = pd.read_csv('data/abalone.data',
                       header=None,
                       names=names)

    enc = preprocessing.LabelBinarizer()
    gender = enc.fit_transform(abalone.Sex)

    y = abalone.Rings.values
    y = y.reshape(-1, 1)

    abalone.drop(['Sex', 'Rings'], axis=1, inplace=True)
    x = np.hstack([gender, abalone.values])

    print(x.shape, y.shape)     # (4177, 10) (4177,)

    return model_selection.train_test_split(x, y, train_size=0.7)


def show_difference(preds, labels):
    preds_1 = preds.reshape(-1)
    y_test_1 = labels.reshape(-1)

    diff = preds_1 - y_test_1
    error = np.mean(np.abs(diff))
    # print('평균 오차 :', error)

    return error


def model_abalone_regression(x_train, x_test, y_train, y_test):
    n_features = x_train.shape[1]
    n_classes = 1

    w = tf.Variable(tf.random_uniform([n_features, n_classes]))
    b = tf.Variable(tf.random_uniform([n_classes]))

    ph_x = tf.placeholder(tf.float32)

    hx = tf.matmul(ph_x, w) + b

    loss_i = (hx - y_train) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    losses, errors = [], []
    for i in range(1001):
        sess.run(train, {ph_x: x_train})

        if i % 10 == 0:
            c = sess.run(loss, {ph_x: x_train})
            print(i, c)

            # ---------------------------- #

            preds_test = sess.run(hx, {ph_x: x_test})
            error = show_difference(preds_test, y_test)

            losses.append(c)
            errors.append(error)

    sess.close()

    plt.plot(range(len(errors)), losses, label='loss')
    plt.plot(range(len(errors)), errors, label='error')
    plt.legend()
    plt.show()


x_train, x_test, y_train, y_test = get_data()
model_abalone_regression(x_train, x_test, y_train, y_test)
