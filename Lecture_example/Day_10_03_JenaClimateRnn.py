# Day_10_03_JenaClimateRnn.py
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# 문제
# jena의 온도 데이터를 RNN으로 모델링해서 평균 오차를 계산하세요


def rnn_jena_temperature():
    jena = pd.read_csv('data/jena_climate_2009_2016.csv', index_col=0)
    jena = jena[:1000]
    degc = jena['T (degC)'].values

    batch_size, seq_length, n_features = 100, 144, 1
    hidden_size = 150

    # --------------------------------- #

    rng = [(i, i+seq_length) for i in range(len(degc) - seq_length)]

    # x = [[degc[s:e]] for s, e in rng]             # error (420407, 1, 144)
    x = [degc[s:e].reshape(-1, 1) for s, e in rng]
    y = [degc[e] for s, e in rng]
    print(np.array(x).shape, np.array(y).shape)     # (420407, 144, 1) (420407,)

    x = np.float32(x)
    y = np.reshape(y, [-1, 1])

    # --------------------------------- #

    ph_x = tf.placeholder(tf.float32, shape=[None, seq_length, n_features])
    ph_y = tf.placeholder(tf.float32)

    cells = [tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size) for _ in range(2)]
    multi = tf.nn.rnn_cell.MultiRNNCell(cells)
    outputs, _states = tf.nn.dynamic_rnn(multi, ph_x, dtype=tf.float32)
    print(outputs.shape)    # (?, 144, 150)

    z = tf.layers.dense(inputs=outputs[:, -1, :], units=1, activation=None)

    loss_i = (z - ph_y) ** 2
    loss = tf.reduce_mean(loss_i)

    # optimizer = tf.train.GradientDescentOptimizer(0.1)
    optimizer = tf.train.AdamOptimizer(0.001)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    epochs = 1
    n_iteration = len(x) // batch_size

    for i in range(epochs):
        total = 0
        for j in range(n_iteration):
            n1 = batch_size * j
            n2 = n1 + batch_size

            xx = x[n1:n2]
            yy = y[n1:n2]

            sess.run(train, {ph_x: xx, ph_y: yy})
            total += sess.run(loss, {ph_x: xx, ph_y: yy})

        print(i, total / n_iteration)

    preds = sess.run(z, {ph_x: x})
    print(preds.shape)

    print('acc :', np.mean(np.abs(preds - y)))
    print('acc :', np.mean(np.abs(preds.reshape(-1) - y.reshape(-1))))
    sess.close()


# 문제 1
# jena 데이터에서 2013년 9월 7일 데이터에서 아무거나 한 개만 출력하세요

# 문제 2
# 2013년 9월 7일 전체 데이터 144개를 출력하세요

# 문제 3
# 인덱스를 날짜/시간으로 변경한 다음 특정 날짜의 데이터를 쉽게 가져와 보세요

def time_series_pandas():
    jena = pd.read_csv('data/jena_climate_2009_2016.csv', index_col=0)
    # jena = jena['T (degC)']
    print(jena.head(), end='\n\n')
    print(jena.index, end='\n\n')

    print(jena.iloc[3])
    print(jena.loc['01.01.2009 00:40:00'])

    # print(jena.iloc[144 * 15])

    print(jena.loc['07.09.2013 00:40:00'])
    print(jena.loc['15.09.2013 00:40:00'])

    print(jena.loc['15.09.2013 00:00:00':'15.09.2013 23:50:00'])

    # print(jena.loc['15.09.2013'])     # error

    print('-' * 30)

    jena.index = pd.to_datetime(jena.index)
    print(jena, end='\n\n')

    print(jena.loc['15.09.2013'])
    print(jena.loc['09.15.2013'])

    print(jena.loc['09.07.2013'])   # 9월 7일
    print(jena.loc['07.09.2013'])   # 7월 9일

    print(jena.loc['2013'])


# 문제
# 아래에 제시한 3개의 컬럼을 사용해서 다음 날의 온도를 예측하세요
# 'p (mbar)', 'rho (g/m**3)', 'T (degC)'
def rnn_jena_multi():
    jena = pd.read_csv('data/jena_climate_2009_2016.csv', index_col=0)

    # jena = jena[:1000]
    jena = jena[['p (mbar)', 'rho (g/m**3)', 'T (degC)']].values
    print(jena)

    batch_size, seq_length, n_features = 100, 144, 3
    hidden_size = 150

    # --------------------------------- #

    rng = [(i, i+seq_length) for i in range(len(jena) - seq_length)]

    x = [jena[s:e] for s, e in rng]
    y = [jena[e][-1] for s, e in rng]
    print(np.array(x).shape, np.array(y).shape)     # (420407, 144, 3) (420407,)

    x = np.float32(x)
    y = np.reshape(y, [-1, 1])

    # --------------------------------- #

    ph_x = tf.placeholder(tf.float32, shape=[None, seq_length, n_features])
    ph_y = tf.placeholder(tf.float32)

    cells = [tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size) for _ in range(2)]
    multi = tf.nn.rnn_cell.MultiRNNCell(cells)
    outputs, _states = tf.nn.dynamic_rnn(multi, ph_x, dtype=tf.float32)
    print(outputs.shape)    # (?, 144, 150)
    print('-' * 30)

    z = tf.layers.dense(inputs=outputs[:, -1, :], units=1, activation=None)

    loss_i = (z - ph_y) ** 2
    loss = tf.reduce_mean(loss_i)

    # optimizer = tf.train.GradientDescentOptimizer(0.1)
    optimizer = tf.train.AdamOptimizer(0.001)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    epochs = 10
    n_iteration = len(x) // batch_size

    for i in range(epochs):
        total = 0
        for j in range(n_iteration):
            n1 = batch_size * j
            n2 = n1 + batch_size

            xx = x[n1:n2]
            yy = y[n1:n2]

            sess.run(train, {ph_x: xx, ph_y: yy})
            total += sess.run(loss, {ph_x: xx, ph_y: yy})

        print(i, total / n_iteration)

    preds = sess.run(z, {ph_x: x})
    print(preds.shape)

    print('acc :', np.mean(np.abs(preds - y)))
    print('acc :', np.mean(np.abs(preds.reshape(-1) - y.reshape(-1))))
    sess.close()


# rnn_jena_temperature()
# time_series_pandas()
rnn_jena_multi()





