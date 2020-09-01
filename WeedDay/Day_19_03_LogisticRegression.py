# Day_19_03_LogisticRegression.py
import tensorflow as tf
import numpy as np
from sklearn import datasets, model_selection
import matplotlib.pyplot as plt


def logistic_regression():
    #       공부  출석
    x = [[1., 1., 2.],  # 탈락
         [1., 2., 1.],
         [1., 4., 5.],  # 통과
         [1., 5., 4.],
         [1., 8., 9.],
         [1., 9., 8.]]
    y = [[0], [0], [1], [1], [1], [1]]
    # y = np.int32(y)
    y = np.float32(y)

    w = tf.Variable(tf.random_uniform([3, 1]))

    # (6, 1) = (6, 3) @ (3, 1)
    z = tf.matmul(x, w)
    hx = tf.sigmoid(z)  # 1 / (1 + tf.exp(-z))

    # loss_i = y * -tf.log(hx) + (1 - y) * -tf.log(1 - hx)
    loss_i = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train)
        print(i, sess.run(loss))

    preds = sess.run(hx)
    preds = preds.reshape(-1)
    print(preds)

    bools = (preds > 0.5)
    print(bools)

    y_bools = y.reshape(-1)
    print(y_bools)

    print('acc :', np.mean(bools == y_bools))

    sess.close()


# 문제
# iris 데이터로부터 2개의 품종을 골라서
# 70%로 학습하고 30%에 대해 정확도를 구하세요
def logistic_regression_iris():
    x, y = datasets.load_iris(return_X_y=True)
    y = y.reshape(-1, 1)
    y = np.float32(y)
    print(x.shape, y.shape)         # (150, 4) (150, 1)
    # print(type(x), type(y))
    # print(x[:3])
    # print(y)

    x_part = x[:100]
    y_part = y[:100]

    data = model_selection.train_test_split(
        x_part, y_part, train_size=0.7)
    x_train, x_test, y_train, y_test = data

    w = tf.Variable(tf.random_uniform([4, 1]))
    b = tf.Variable(tf.random_uniform([1]))

    ph_x = tf.placeholder(tf.float32)

    # (100, 1) = (100, 4) @ (4, 1)
    z = tf.matmul(ph_x, w) + b
    hx = tf.sigmoid(z)

    loss_i = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=y_train, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train, {ph_x: x_train})
        print(i, sess.run(loss, {ph_x: x_train}))

    preds = sess.run(hx, {ph_x: x_test})
    preds = preds.reshape(-1)
    print(preds)

    bools = np.int32(preds > 0.5)
    print(bools)

    y_bools = np.int32(y_test.reshape(-1))
    print(y_bools)

    print('acc :', np.mean(bools == y_bools))
    sess.close()

    plt.scatter(x_part[:, 0], x_part[:, 2],
                c=y_part.reshape(-1))
    plt.show()


# logistic_regression()
logistic_regression_iris()

