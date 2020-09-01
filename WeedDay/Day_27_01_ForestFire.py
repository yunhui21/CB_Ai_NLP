# Day_27_01_ForestFire.py
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import model_selection, preprocessing
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=1000)


# 문제 1
# 산불 데이터에 대해 오차를 구하세요

# 문제 2
# loss와 error를 각각의 그래프에 그리세요
# train과 test를 함께 그립니다

def get_data():
    fires = pd.read_csv('data/forestfires.csv')
    print(fires)

    fires.info()
    print(fires.describe())

    enc = preprocessing.LabelEncoder()
    month = enc.fit_transform(fires.month)
    month = month.reshape(-1, 1)
    day = enc.fit_transform(fires.day)
    day = day.reshape(-1, 1)

    area = enc.fit_transform(fires.area)
    y = area.reshape(-1, 1)

    fires.drop(['month', 'day', 'area'], axis=1, inplace=True)
    x = np.hstack([month, day, fires.values])

    print(x.shape, y.shape)     # (517, 12) (517, 1)

    return model_selection.train_test_split(x, y, train_size=0.7)


def show_difference(preds, labels):
    preds_1 = preds.reshape(-1)
    y_test_1 = labels.reshape(-1)

    diff = preds_1 - y_test_1
    error = np.mean(np.abs(diff))
    # print('평균 오차 :', error)

    return error


def show_plot(idx, values1, title1, values2, title2):
    plt.subplot(1, 2, idx)
    plt.plot(range(len(values1)), values1, label=title1)
    plt.plot(range(len(values2)), values2, label=title2)
    plt.legend()


def model_forest_fires_regression(x_train, x_test, y_train, y_test):
    n_features = x_train.shape[1]
    n_classes = 1

    w = tf.Variable(tf.random_uniform([n_features, n_classes]))
    b = tf.Variable(tf.random_uniform([n_classes]))

    ph_x = tf.placeholder(tf.float32)
    ph_y = tf.placeholder(tf.float32)

    hx = tf.matmul(ph_x, w) + b

    loss_i = (hx - ph_y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.AdamOptimizer(0.01)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    losses_train, losses_test = [], []
    errors_train, errors_test = [], []

    train_set = {ph_x: x_train, ph_y: y_train}
    test_set = {ph_x: x_test, ph_y: y_test}
    for i in range(101):
        sess.run(train, train_set)

        if i % 10 == 0:
            c_train = sess.run(loss, train_set)
            c_test = sess.run(loss, test_set)

            preds_train = sess.run(hx, train_set)
            preds_test = sess.run(hx, test_set)

            error_train = show_difference(preds_train, y_train)
            error_test = show_difference(preds_test, y_test)

            losses_train.append(c_train)
            losses_test.append(c_test)

            errors_train.append(error_train)
            errors_test.append(error_test)

            print('{:5} : {:11.4f}  {:9.4f}'.format(i, c_test, error_test))

    sess.close()

    show_plot(1,
              losses_train, 'train loss',
              losses_test, 'test loss')
    show_plot(2,
              errors_train, 'train error',
              errors_test, 'test error')
    plt.show()


x_train, x_test, y_train, y_test = get_data()
model_forest_fires_regression(x_train, x_test, y_train, y_test)
