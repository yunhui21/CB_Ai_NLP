# Day_10_01_ligisticregressioin.py

import tensorflow as tf

# y = x1 +  x2
# y = 1 * x1 + 2 * x2 + 0

x1 = [1, 0, 3, 0, 5]       # 공부한 시간
x2 = [0, 2, 0, 4, 0]       # 출석한 일수
y  = [1, 2, 3, 4, 5]       # 성적

# 문제
# w, b 를 만들고 수식을 완성하세요.
def multifle_regression_1():
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

def multifle_regression_2():

# multifle_regression_1()
multifle_regression_2()

