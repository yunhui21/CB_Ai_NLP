# Day_10_01_RnnMnist.py
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf

# 문제
# mnist를 rnn 모델로 풀어보세요 (미니배치 방식)


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


def rnn_mnist_minibatch():
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
    hidden_size = 150
    batch_size = 100

    ph_x = tf.placeholder(tf.float32, shape=[None, seq_length, n_features])
    ph_y = tf.placeholder(tf.int32)

    cells = [tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size) for _ in range(2)]
    multi = tf.nn.rnn_cell.MultiRNNCell(cells)
    outputs, _states = tf.nn.dynamic_rnn(multi, ph_x, dtype=tf.float32)

    z = tf.layers.dense(inputs=outputs[:, -1, :], units=n_classes, activation=None)
    hx = tf.nn.softmax(z)

    loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=z, labels=ph_y)
    loss = tf.reduce_mean(loss_i)

    # optimizer = tf.train.GradientDescentOptimizer(0.1)
    optimizer = tf.train.AdamOptimizer(0.001)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    epochs = 10
    n_iteration = len(x_train) // batch_size

    for i in range(epochs):
        total = 0
        for j in range(n_iteration):
            n1 = batch_size * j
            n2 = n1 + batch_size

            xx = x_train[n1:n2]
            yy = y_train[n1:n2]

            sess.run(train, {ph_x: xx, ph_y: yy})
            total += sess.run(loss, {ph_x: xx, ph_y: yy})

        print(i, total / n_iteration)

    preds = sess.run(z, {ph_x: x_test})
    preds_arg = np.argmax(preds, axis=1)

    print('acc :', np.mean(preds_arg == y_test))
    sess.close()


# rnn_mnist()
rnn_mnist_minibatch()

