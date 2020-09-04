# Day_10_01_RnnMnist.py
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf
# 문제
# mnist를 rnn 모델로 풀어보세요(미니배치 방식)


def rnn_mnist():
    mnist = input_data.read_data_sets('mnist', one_hot=False)

    x_train = np.vstack([mnist.train.images, mnist.validation.images])  # vstack 다차원
    y_train = np.concatenate([mnist.train.labels, mnist.validation.labels])  # 그냥 연결하기면됨 concat
    x_test  = mnist.test.images
    y_test  = mnist.test.labels

    # print(type(x_train))
    # print(type(y_train))

    print(x_train.shape, x_test.shape)  # (60000, 784) (10000, 784)
    print(y_train.shape, y_test.shape)  # (60000,) (10000,)

    # preprocessing을 해야 하는가 말아야 하는가?
    # x_train과 y_train을 함께 mnist로 preprocessing 하는가?
    # x_train과 y_train을 따로 preprocessing 하는가?
    # preprocessing의 정확한 의미를 파악해라.(이미지와 텍스트의 차이점이 있는지 구분해서)
    # seq_length와 n_features의 값은? shape을 통해서 n_features는 784로 보여진다.
    # 너무 어렵다. ...
    # 3차원으로 데이터 변환

    x_train = x_train.reshape(-1, 28, 28)   # 앞의  28 seq_length
    x_test  = x_test.reshape(-1, 28, 28)    # 뒤의 28 feares개수
    y_train = np.int32(y_train)
    y_test = np.int32(y_test)   # error int32, int64로 넣어야 한다.

    batch_size, seq_length, n_features = x_train.shape   # 값을 알아야겠지 2차원데이터를 갖고 있음.. Rnn 3차원 필요.
    n_classes = 10 # 분류의 문제이므로 dense레이어에서 unit값이 된다.
    hidden_size = 9     # train의 개수 6만개이므로 작업하고 나서 수정이 필요하면 한다.

    ph_x = tf.placeholder(tf.float32, shape=[None, seq_length, n_features]) #

    cells = [tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size) for _ in range(2)]
    multi = tf.nn.rnn_cell.MultiRNNCell(cells)
    outputs, _states = tf.nn.dynamic_rnn(multi, ph_x, dtype=tf.float32)

    z = tf.layers.dense(inputs=outputs[:, -1, :], units=n_classes, activation=None)
    hx = tf.nn.softmax(z)

    loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(logits= z, labels = y_train)
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

    x_train = np.vstack([mnist.train.images, mnist.validation.images])  # vstack 다차원
    y_train = np.concatenate([mnist.train.labels, mnist.validation.labels])  # 그냥 연결하기면됨 concat
    x_test  = mnist.test.images
    y_test  = mnist.test.labels

    # print(type(x_train))
    # print(type(y_train))

    print(x_train.shape, x_test.shape)  # (60000, 784) (10000, 784)
    print(y_train.shape, y_test.shape)  # (60000,) (10000,)

    # preprocessing을 해야 하는가 말아야 하는가?
    # x_train과 y_train을 함께 mnist로 preprocessing 하는가?
    # x_train과 y_train을 따로 preprocessing 하는가?
    # preprocessing의 정확한 의미를 파악해라.(이미지와 텍스트의 차이점이 있는지 구분해서)
    # seq_length와 n_features의 값은? shape을 통해서 n_features는 784로 보여진다.
    # 너무 어렵다. ...
    # 3차원으로 데이터 변환

    x_train = x_train.reshape(-1, 28, 28)   # 앞의  28 seq_length
    x_test  = x_test.reshape(-1, 28, 28)    # 뒤의 28 feares개수
    y_train = np.int32(y_train)
    y_test = np.int32(y_test)   # error int32, int64로 넣어야 한다.

    batch_size, seq_length, n_features = x_train.shape   # 값을 알아야겠지 2차원데이터를 갖고 있음.. Rnn 3차원 필요.
    n_classes = 10 # 분류의 문제이므로 dense레이어에서 unit값이 된다.
    hidden_size = 150  # train의 개수 6만개이므로 작업하고 나서 수정이 필요하면 한다.
    batch_size = 100    # 방향을 잡아야 할때

    ph_x = tf.placeholder(tf.float32, shape=[None, seq_length, n_features]) #
    ph_y = tf.placeholder(tf.int32) # 내부적으로 숨겨서 보이지 않을것이다.

    cells = [tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size) for _ in range(2)]
    multi = tf.nn.rnn_cell.MultiRNNCell(cells)
    outputs, _states = tf.nn.dynamic_rnn(multi, ph_x, dtype=tf.float32)

    z = tf.layers.dense(inputs=outputs[:, -1, :], units=n_classes, activation=None)
    hx = tf.nn.softmax(z)

    loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(logits= z, labels = ph_y)
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

            sess.run(train, {ph_x: xx, ph_y:yy})
            total = sess.run(loss, {ph_x: xx, ph_y:yy})

        print(i, total / n_iteration)

    preds = sess.run(z, {ph_x: x_test})
    preds_arg = np.argmax(preds, axis=1)

    print('acc:', np.mean(preds_arg == y_test))
    sess.close()

rnn_mnist_minibatch()

