# Day_17_01_LinearRegression.py
import tensorflow as tf         # 1.14.0
import numpy as np


def linear_regression_1():
    x = [1, 2, 3]
    y = [1, 2, 3]

    w = tf.Variable(5.0)
    b = tf.Variable(-3.0)

    hx = w * x + b
    # hx = tf.add(tf.multiply(w, x), b)

    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(loss=loss)
    # train = tf.train.GradientDescentOptimizer(0.1).minimize(tf.reduce_mean((tf.add(tf.multiply(w, x), b) - y) ** 2))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train)
        print(sess.run(loss))

    # print(w)
    # print(sess.run(w))

    # 문제
    # x가 5와 7일 때의 결과를 알려주세요 (2가지)
    # 1. 계산식을 직접 구성
    print(w * x + b)
    print(sess.run(w))
    print(sess.run(b))
    print(sess.run(hx))
    print(sess.run(loss_i))
    print(sess.run(loss))
    print(sess.run(train))
    print(sess.run(w * x + b))
    print(sess.run(w * [1, 2, 3] + b))
    print('-' * 30)

    print('5 :', sess.run(w * 5 + b))
    print('7 :', sess.run(w * 7 + b))
    print('* :', sess.run(w * [5, 7] + b))

    # 2. 계산의 결과를 이용
    ww, bb = sess.run(w), sess.run(b)
    print(ww, bb)
    print('5 :', ww * 5 + bb)
    print('7 :', ww * 7 + bb)

    sess.close()


def linear_regression_2():
    x = [1, 2, 3]
    y = [1, 2, 3]

    # w = tf.Variable(np.float32(np.random.rand(1)))
    # b = tf.Variable(np.float32(np.random.rand(1)))

    w = tf.Variable(tf.random_uniform([1]))
    b = tf.Variable(tf.random_uniform([1]))

    ph_x = tf.placeholder(tf.float32)
    ph_y = tf.placeholder(tf.float32)

    hx = w * ph_x + b

    loss_i = (hx - ph_y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train, feed_dict={ph_x: x, ph_y: y})
        print(i, sess.run(loss, {ph_x: x, ph_y: y}))

    # 문제
    # x가 5와 7일 때의 결과를 알려주세요
    print(sess.run(hx, {ph_x: x, ph_y: y}))

    sess.close()


def linear_regression_3():
    x = [1, 2, 3]
    y = [1, 2, 3]

    w = tf.Variable(tf.random_uniform([1]))
    b = tf.Variable(tf.random_uniform([1]))

    ph_x = tf.placeholder(tf.float32)
    # ph_y = tf.placeholder(tf.float32)

    hx = w * ph_x + b

    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train, {ph_x: x})
        print(i, sess.run(loss, {ph_x: x}))

    # 문제
    # x가 5와 7일 때의 결과를 알려주세요
    print(sess.run(hx, {ph_x: x}))
    print(sess.run(hx, {ph_x: [5, 7]}))

    sess.close()


# linear_regression_1()
# linear_regression_2()
# linear_regression_3()
