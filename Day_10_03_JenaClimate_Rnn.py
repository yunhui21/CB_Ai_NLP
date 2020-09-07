# Day_10_03_JenaClimateRnn.py
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import datetime

# 문제
# jena의 온도 데이터를 RNN으로 모델링해서 평균 오차를 계산하세요

def rnn_mnist():
    mnist = input_data.read_data_sets('mnist', one_hot=False)

    x_train = np.vstack([mnist.train.images, mnist.validation.images])
    y_train = np.concatenate([mnist.train.labels, mnist.validation.labels])
    x_test = mnist.test.images
    y_test = mnist.test.labels

    print(x_train.shape, x_test.shape)  # (60000, 784) (10000, 784)
    print(y_train.shape, y_test.shape)  # (60000,) (10000,)

    x_train = x_train.reshape(-1, 28, 28)
    x_test = x_test.reshape(-1, 28, 28)
    y_train = np.int32(y_train)
    y_test = np.int32(y_test)

    batch_size, seq_length, n_features = x_train.shape      # (60000, 28, 28)
    n_classes = 10
    hidden_size = 9

    ph_x = tf.placeholder(tf.float32, shape=[None, seq_length, n_features])

    cells = [tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size) for _ in range(2)]
    multi = tf.nn.rnn_cell.MultiRNNCell(cells)
    outputs, _states = tf.nn.dynamic_rnn(multi, ph_x, dtype=tf.float32)

    z = tf.layers.dense(inputs=outputs[:, -1, :], units=n_classes, activation=None)
    hx = tf.nn.softmax(z)

    loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=z, labels=y_train)
    loss = tf.reduce_mean(loss_i)

    # optimizer = tf.train.GradientDescentOptimizer(0.1)
    optimizer = tf.train.AdamOptimizer(0.01)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1):
        sess.run(train, {ph_x: x_train})
        c = sess.run(loss, {ph_x: x_train})
        print(i, c)

    preds = sess.run(z, {ph_x: x_test})
    sess.close()


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
    n_iteration = len(x) // batch_size  #둘을 곱하면 전체 데이터의 개수

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
    print(preds.shape)  #x,y모두 2차원이므로 연산은 문제 없지만

    print('acc :', np.mean(np.abs(preds - y)))  #
    print('acc :', np.mean(np.abs(preds.reshape(-1) - y.reshape(-1))))  #
    sess.close()

# 문제 1
# jena 데이터에서 2013년 9월 7일 데이터를 출력하세요.

# 문제 2
# 2013년 9월 7일 전체 데이터를 출력하세요.

# 문제3
# 인덱스를 날짜/시간으로 변경한 다음 특정 날짜의 데이터를 쉽게 가져와 보세요.
# index to datetime

# 문제 4
# 아래의 제시한 3개의 컬럼을 사용해서 다음 날의 온도를 예측하시요.
# 'p (mbar)','rho (g/m**3)', 'T (degC)'

def time_series_pandas():
    jena = pd.read_csv('data/jena_climate_2009_2016.csv', index_col=0)
    jena = jena['T (degC)']
    print(jena.head(), end='\n\n')
    print(jena.index, end='\n\n')

    # 요기에 출력하세요.
    # 날짜와 시간으로만 표기/ 년도를 삭제. 전체 데이터 변환

    print(jena.iloc[144 * 15])

    print(jena.loc['01.01.2009 00:40:00'])
    print(jena.loc['15.09.2013 00:10:00'])

    print(jena.loc['07.09.2013 00:00:00':'07.09.2013 23:50:00'])

    jena.index = pd.to_datetime(jena.index)
    print(jena, end='\n\n')

    print(jena.loc['2013-09-07'])
    print(jena.loc['07.09.2013'])
    print(jena.loc['2013'])


def rnn_jena_multi():
    jena = pd.read_csv('data/jena_climate_2009_2016.csv', index_col=0)
    print(jena.columns)
    # jena = jena[:1000]
    jena = jena[['p (mbar)','rho (g/m**3)', 'T (degC)']].values
    print(jena)
    # degc = jena['T (degC)'].values
    # print(jena.shape)  # 1000, 3


    batch_size, seq_length, n_features = 100, 144, 3
    hidden_size = 150

    # --------------------------------- #

    rng = [(i, i+seq_length) for i in range(len(jena) - seq_length)]

    # x = [[degc[s:e]] for s, e in rng]             # error (420407, 1, 144)
    x = [jena[s:e] for s, e in rng]     #reshape과정이 필요 없음
    y = [jena[e][-1] for s, e in rng]   #(856, 3) 3개의 결과치 나오는 결과..
    print(np.array(x).shape, np.array(y).shape)     # (856, 144, 3) (856, 3)

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

    epochs = 10
    n_iteration = len(x) // batch_size  #둘을 곱하면 전체 데이터의 개수

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
    print(preds.shape)  #x,y모두 2차원이므로 연산은 문제 없지만

    print('acc :', np.mean(np.abs(preds - y)))  #
    print('acc :', np.mean(np.abs(preds.reshape(-1) - y.reshape(-1))))  #
    sess.close()



# rnn_jena_temperature()
# time_series_pandas()
rnn_jena_multi()





























# # Day_10_03_JenaClimate_Rnn.py
# from tensorflow.examples.tutorials.mnist import input_data
# from tensorflow.examples.tutorials.mnist import input_data
# import numpy as np
# import tensorflow as tf
# import pandas as pd
# from sklearn import preprocessing, model_selection
# # 문제
# # 10_01_rnn_Mnist_minibatch
# # jena의 온도 데이터를 RNN으로 모델링해서 평균 오차를 계산하세요.
#
# def rnn_jena_temperature():
#     jena = pd.read_csv('data/jena_climate_2009_2016.csv', index_col=0)
#     jena = jena[:1000]
#     degc = jena['T (degC)'].values
#
#     batch_size, seq_length, n_features = 100, 144, 1
#     hidden_size = 150  # train의 개수 6만개이므로 작업하고 나서 수정이 필요하면 한다.
#
#     rng = [(i, i + seq_length) for i in range(len(degc) - seq_length)]
#
#     # x = [[degc[s:e]] for s, e in rng] #(420407, 1, 144)
#     x = [degc[s:e].reshape(-1,1) for s, e in rng]   # (420407, 144, 1)
#     y = [degc[e] for s, e in rng]
#     print(np.array(x).shape, np.array(y).shape) #(420407, 144)->(420407, 144, 1) (420407,)
#     # return
#     x = np.float32([x])
#     y = np.reshape(y, [-1, 1])
# #_________________________#
#     ph_x = tf.placeholder(tf.float32, shape=[None, seq_length, n_features]) #
#     ph_y = tf.placeholder(tf.float32) # 내부적으로 숨겨서 보이지 않을것이다.
#
#     cells = [tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size) for _ in range(2)]
#     multi = tf.nn.rnn_cell.MultiRNNCell(cells)
#     outputs, _states = tf.nn.dynamic_rnn(multi, ph_x, dtype=tf.float32)
#     # print(outputs.shape) # (?, 144, 150)
#
#     z = tf.layers.dense(inputs=outputs[:, -1, :], units=1, activation=None)
#
#     loss_i = (z - ph_y) ** 2
#     loss = tf.reduce_mean(loss_i)
#
#     # optimizer = tf.train.GradientDescentOptimizer(0.1)
#     optimizer = tf.train.AdamOptimizer(0.001)
#     train = optimizer.minimize(loss)
#
#     sess = tf.Session()
#     sess.run(tf.global_variables_initializer())
#
#     epochs = 10
#     n_iteration = len(x) // batch_size
#
#     for i in range(epochs):
#         total = 0
#         for j in range(n_iteration):
#             n1 = batch_size * j
#             n2 = n1 + batch_size
#
#             xx = x[n1:n2]
#             yy = y[n1:n2]
#
#             sess.run(train, {ph_x: xx, ph_y:yy})
#             total += sess.run(loss, {ph_x: xx, ph_y:yy})
#
#         print(i, total / n_iteration)
#
#     preds = sess.run(z, {ph_x: x})
#     print(preds.shape)
#     print('acc:', np.mean(preds == y))
#     sess.close()
#
# rnn_jena_temperature()