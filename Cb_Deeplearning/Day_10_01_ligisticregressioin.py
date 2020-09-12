# Day_10_01_ligisticregressioin.py

import tensorflow as tf
import numpy as np

# 문제
# w, b 를 만들고 수식을 완성하세요.
#
def multiple_regression_1():
    # y = x1 +  x2
    # y = 1 * x1 + 2 * x2 + 0

    x1 = [1, 0, 3, 0, 5]  # 공부한 시간
    x2 = [0, 2, 0, 4, 0]  # 출석한 일수
    y = [1, 2, 3, 4, 5]  # 성적

    w1 = tf.Variable(tf.random_uniform([1]))     # 60%
    w2 = tf.Variable(tf.random_uniform([1]))
    b  = tf.Variable(tf.random_uniform([1]))     # 40%

    hx = w1 * x1 + w2 * x1 + b

    # 정답까지의 거리 loss
    loss_1 = (hx - y) ** 2
    loss = tf.reduce_mean(loss_1)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for  i in range(10):
        sess.run(train)
        print(i, sess.run(loss))

    sess.close()

# 문제
# 피쳐를 변수 1개로 통합하세요.
# X만 마구지 않고 w도 바꿔야 한다.

# def multiple_regression_2():
#     x = [[1, 0, 3, 0, 5],  # 공부한 시간
#           [0, 2, 0, 4, 0]]  # 출석한 일수
#     y = [1, 2, 3, 4, 5]  # 성적
#
#     w = tf.Variable(tf.random.uniform([2]))     # 60%
#     b = tf.Variable(tf.random,uniform([1]))     # 40%
#
#     hx = w[0] * x[0] +w[1] * x[1]+ b
#
#     # 정답까지의 거리 loss
#     loss_1 = (hx - y) ** 2
#     loss = tf.reduce_mean(loss_1)
#
#     optimizer = tf.train.GradientDescentOptimizer(0.1)
#     train = optimizer.minimize(loss)
#
#     sess = tf.Session()
#     sess.run(tf.global_variables_initializer())
#
#     for  i in range(10):
#         sess.run(train)
#         print(i, sess.run(loss))
#
#     sess.close()

# 문제
# bias를   통합하세요.

def multiple_regression_3():
    # x = [[1, 0, 3, 0, 5],  # 공부한 시간
    #      [0, 2, 0, 4, 0],
    #      [1, 1, 1, 1, 1]]  # 출석한 일수 weight의 값이 다른건 다루는 피쳐가 다르므로 피쳐가 가지는
    # x = [[1, 0, 3, 0, 5],
    #      [1, 1, 1, 1, 1],
    #      [0, 2, 0, 4, 0]]
    x = [[1, 1, 1, 1, 1],
         [1, 0, 3, 0, 5],
         [0, 2, 0, 4, 0]]       # 바이어스가 보이도록 위로 올려놓는다.
    y = [1, 2, 3, 4, 5]  # 성적

    # x = np.reshape([x])

    w = tf.Variable(tf.random.uniform([3]))     # 60%

    # hx = w[0] * x[0] + w[1] * x[1] + b
    # hx = w[0] * x[0] + w[1] * x[1] + w[2]
    # hx = w[0] * x[0] + w[1] * x[1] + w[2] * 1    # brodcast연산
    hx = w[0] * x[0] + w[1] * x[1] + w[2] * x[2]    # brodcast연산

    # 정답까지의 거리 loss
    loss_1 = (hx - y) ** 2
    loss = tf.reduce_mean(loss_1)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for  i in range(100):
        sess.run(train)
        print(i, sess.run(loss))
    print(sess.run(w))
    sess.close()

# 문제
# hx를 행렬 곱셈으로 바꾸세요.(tf.matmul)

def multiple_regression_4():
    # x = [[1., 1., 1., 1., 1.],       #
    #      [1., 0., 3., 0., 5.],
    #      [0., 2., 0., 4., 0.]]       # 바이어스가 보이도록 위로 올려놓는다.
    x = [[1., 1., 1., 1., 1.],       #
         [1., 0., 3., 0., 5.],
         [0., 2., 0., 4., 0.]]       # 바이어스가 보이도록 위로 올려놓는다.

    y = [1, 2, 3, 4, 5]  # 성적

    # x = np.float32(x)
    # (1, 5) = (1, 3) @ (3, 5) 행렬의 곱은 col과 row의 값이 같아야 한다.
    w = tf.Variable(tf.random.uniform([1,3]))     # 60% 2차원
    # hx = w[0] * x[0] + w[1] * x[1] + w[2] * x[2]    # brodcast연산 , 행렬연산
    hx = tf.matmul(w, x)

    # 정답까지의 거리 loss
    loss_1 = (hx - y) ** 2
    loss = tf.reduce_mean(loss_1)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for  i in range(100):
        sess.run(train)
        print(i, sess.run(loss))
    print(sess.run(hx)) # (1, 5)
    sess.close()


# 문제
# 3시간 공부하고 5번 출석한 학생과 4시간 공부하고 두번 출석한 학생을 구하세요.(palceholder)

def multiple_regression_5():
    x = [[1., 1., 1., 1., 1.],       #
         [1., 0., 3., 0., 5.],
         [0., 2., 0., 4., 0.]]       # 바이어스가 보이도록 위로 올려놓는다.

    y = [1, 2, 3, 4, 5]  # 성적

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
    print(sess.run(hx, {ph_x: [[1., 1., 1., 1., 1.],       #
                               [1., 0., 3., 0., 5.],
                               [0., 2., 0., 4., 0.]]}))
    print(sess.run(hx, {ph_x: [[1., 1.],  #
                               [3., 4.],
                               [5., 2.]]}))
    sess.close()


# multiple_regression_1()
# multiple_regression_2()
# multiple_regression_3()
multiple_regression_4()
# multiple_regression_5()


# a = np.arange(5)
# print(a)
#
# b = 1
# c = [1, 1, 1, 1, 1]
# print(a + b)        # broadcast (배열과 스칼라)
# print(a + c)        # vecter


