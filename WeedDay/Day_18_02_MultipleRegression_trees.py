# Day_18_02_MultipleRegression_trees.py


# 문제
# trees.csv 파일을 읽어서
# Girth가 10이고 Height가 70일 때와
# Girth가 20이고 Height가 80일 때의 Volume을 구해보세요

# Day_18_01_MultipleRegression.py
import tensorflow as tf
import pandas as pd
import numpy as np


def get_trees_wx():
    trees = pd.read_csv('data/trees.csv', index_col=0)
    # print(trees)

    girth = trees.Girth.values
    height = trees.Height.values
    volume = trees.Volume.values

    return np.float32([[1] * len(girth), girth, height]), volume


def get_trees_xw():
    # x, y = get_trees_wx()
    # return x.transpose(), y.reshape(-1, 1)

    trees = pd.read_csv('data/trees.csv', index_col=0)
    # print(trees)

    girth = trees.Girth.values.reshape(-1, 1)
    height = trees.Height.values.reshape(-1, 1)
    volume = trees.Volume.values.reshape(-1, 1)
    bias = np.ones([len(girth), 1], dtype=np.float32)

    return np.hstack([bias, girth, height]), volume


def multiple_regression_5():
    x, y = get_trees_wx()
    # print(x.shape, y.shape)     # (3, 31) (31,)

    # x = [[1., 1., 1., 1., 1.],
    #      [1., 0., 3., 0., 5.],
    #      [0., 2., 0., 4., 0.]]
    # y = [1, 2, 3, 4, 5]

    w = tf.Variable(tf.random_uniform([1, 3]))
    ph_x = tf.placeholder(tf.float32)

    # (1, 5) = (1, 3) @ (3, 5)
    hx = tf.matmul(w, ph_x)

    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.0001)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train, {ph_x: x})
        print(i, sess.run(loss, {ph_x: x}))

    print(sess.run(hx, {ph_x: [[ 1.,  1.],
                               [10., 20.],
                               [70., 80.]]}))
    sess.close()


def multiple_regression_6():
    x, y = get_trees_xw()
    # x = [[1., 1., 0.],
    #      [1., 0., 2.],
    #      [1., 3., 0.],
    #      [1., 0., 4.],
    #      [1., 5., 0.]]
    # y = [[1],
    #      [2],
    #      [3],
    #      [4],
    #      [5]]

    w = tf.Variable(tf.random_uniform([3, 1]))
    ph_x = tf.placeholder(tf.float32)

    # (5, 1) = (5, 3) @ (3, 1)
    hx = tf.matmul(ph_x, w)

    #     (5, 1) - (1, 5)
    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.0001)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train, {ph_x: x})
        print(i, sess.run(loss, {ph_x: x}))

    print(sess.run(hx, {ph_x: [[1., 10., 70.],
                               [1., 20., 80.]]}))
    sess.close()


# multiple_regression_5()
multiple_regression_6()



