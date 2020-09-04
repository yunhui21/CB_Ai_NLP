# Day_09_02_Stock.py
import tensorflow as tf
import numpy as np
from sklearn import preprocessing, model_selection
import matplotlib.pyplot as plt
import pandas as pd

# 문제 1
# 7일분의 데이터로 다음 날 종가를 예측하는 모델을 만드세요

# 문제 2
# 주식 데이터를 70%로 학습하고 30%로 예측하세요


def rnn_stock_1():
    # stock = pd.read_csv('data/stock_daily.csv')
    stock = np.loadtxt('data/stock_daily.csv', delimiter=',')
    stock = stock[::-1]
    stock = preprocessing.minmax_scale(stock)
    print(stock.shape)          # (732, 5)

    seq_length, n_features = 7, 5
    hidden_size = 9

    rng = [(i, i+seq_length) for i in range(len(stock) - seq_length)]
    print(rng[-1])              # (724, 731)

    x = [stock[s:e] for s, e in rng]
    y = [stock[e][-1] for s, e in rng]
    print(y[-1])
    print(np.array(x).shape, np.array(y).shape)     # (725, 7, 5) (725,)

    # for i, j in zip(x, y):
    #     print(i[-1], j)

    x = np.float32(x)
    y = np.reshape(y, [-1, 1])

    cells = [tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size) for _ in range(2)]
    multi = tf.nn.rnn_cell.MultiRNNCell(cells)
    outputs, _states = tf.nn.dynamic_rnn(multi, x, dtype=tf.float32)

    z = tf.layers.dense(inputs=outputs[:, -1, :], units=1, activation=None)
    print(z.shape)          # (725, 1)

    loss_i = (z - y) ** 2
    loss = tf.reduce_mean(loss_i)

    # optimizer = tf.train.GradientDescentOptimizer(0.1)
    optimizer = tf.train.AdamOptimizer(0.01)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train)
        c = sess.run(loss)
        print(i, c)

    # 문제
    # 정답과 예측한 결과를 그래프로 그려보세요
    preds = sess.run(z)
    sess.close()

    idx = range(len(y))
    plt.plot(idx, y, 'r')
    plt.plot(idx, preds, 'g')
    plt.show()


def rnn_stock_2():
    # stock = pd.read_csv('data/stock_daily.csv')
    stock = np.loadtxt('data/stock_daily.csv', delimiter=',')
    stock = stock[::-1]
    stock = preprocessing.minmax_scale(stock)
    print(stock.shape)          # (732, 5)

    seq_length, n_features = 7, 5
    hidden_size = 9

    rng = [(i, i+seq_length) for i in range(len(stock) - seq_length)]
    print(rng[-1])              # (724, 731)

    x = [stock[s:e] for s, e in rng]
    y = [stock[e][-1] for s, e in rng]
    print(y[-1])
    print(np.array(x).shape, np.array(y).shape)     # (725, 7, 5) (725,)

    x = np.float32(x)
    y = np.reshape(y, [-1, 1])

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.7, shuffle=False)
    ph_x = tf.placeholder(tf.float32, shape=[None, seq_length, n_features])

    cells = [tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size) for _ in range(2)]
    multi = tf.nn.rnn_cell.MultiRNNCell(cells)
    outputs, _states = tf.nn.dynamic_rnn(multi, ph_x, dtype=tf.float32)

    z = tf.layers.dense(inputs=outputs[:, -1, :], units=1, activation=None)

    loss_i = (z - y_train) ** 2
    loss = tf.reduce_mean(loss_i)

    # optimizer = tf.train.GradientDescentOptimizer(0.1)
    optimizer = tf.train.AdamOptimizer(0.01)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train, {ph_x: x_train})
        c = sess.run(loss, {ph_x: x_train})
        print(i, c)

    # 문제
    # 정답과 예측한 결과를 그래프로 그려보세요
    preds = sess.run(z, {ph_x: x_test})
    sess.close()

    idx = range(len(y_test))
    plt.plot(idx, y_test, 'r')
    plt.plot(idx, preds, 'g')
    plt.show()


# rnn_stock_1()
rnn_stock_2()

#             r5
# r1 r2 r3 r4 r5
# ^  ^  ^  ^  ^
# d1 d2 d3 d4 d5








