# Day_10_02_Multiregreeesion.py
import tensorflow as tf
import numpy as np
import pandas as pd

# 문제
# trees.csv 파일로부터 x와 y데이터를 반환하는 함수를 만드세요.

def get_data():
    f = open('../data/trees.csv', 'r')



def multiple_regression_trees_5():

    w = tf.Variable(tf.random.uniform([1,3]))     # 60% 2차원

    ph_x = tf.placeholder(np.float32) # 함수의 매겨변수와 같은 형태

    hx = tf.matmul(w, ph_x)

    # 정답까지의 거리 loss
    loss_1 = (hx - y) ** 2
    loss = tf.reduce_mean(loss_1)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for  i in range(100):
        sess.run(train, {ph_x: x})
        print(i, sess.run(loss, {ph_x: x}))

    print(sess.run(hx, {ph_x: x}))
    # print(sess.run(hx, {ph_x: [[1., 1., 1., 1., 1.],       #
    #                            [1., 0., 3., 0., 5.],
    #                            [0., 2., 0., 4., 0.]]}))
    # print(sess.run(hx, {ph_x: [[1., 1.],  #
    #                            [3., 4.],
    #                            [5., 2.]]}))
    sess.close()

multiple_regression_trees_5()