# Day_10_03_JenaClimate_Rnn.py
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn import preprocessing, model_selection
# 문제
# 10_01_rnn_Mnist_minibatch
# jena의 온도 데이터를 RNN으로 모델링해서 평균 오차를 계산하세요.

def rnn_jena_temperature():
    jena = pd.read_csv('data/jena_climate_2009_2016.csv', index_col=0)
    jena = jena[:1000]
    degc = jena['T (degC)'].values

    batch_size, seq_length, n_features = 100, 144, 1
    hidden_size = 150  # train의 개수 6만개이므로 작업하고 나서 수정이 필요하면 한다.

    rng = [(i, i + seq_length) for i in range(len(degc) - seq_length)]

    # x = [[degc[s:e]] for s, e in rng] #(420407, 1, 144)
    x = [degc[s:e].reshape(-1,1) for s, e in rng]   # (420407, 144, 1)
    y = [degc[e] for s, e in rng]
    print(np.array(x).shape, np.array(y).shape) #(420407, 144)->(420407, 144, 1) (420407,)
    # return
    x = np.float32([x])
    y = np.reshape(y, [-1, 1])
#_________________________#
    ph_x = tf.placeholder(tf.float32, shape=[None, seq_length, n_features]) #
    ph_y = tf.placeholder(tf.float32) # 내부적으로 숨겨서 보이지 않을것이다.

    cells = [tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size) for _ in range(2)]
    multi = tf.nn.rnn_cell.MultiRNNCell(cells)
    outputs, _states = tf.nn.dynamic_rnn(multi, ph_x, dtype=tf.float32)
    # print(outputs.shape) # (?, 144, 150)
    return

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

            sess.run(train, {ph_x: xx, ph_y:yy})
            total += sess.run(loss, {ph_x: xx, ph_y:yy})

        print(i, total / n_iteration)

    preds = sess.run(z, {ph_x: x})
    print(preds.shape)
    print('acc:', np.mean(preds == y))


rnn_jena_temperature()