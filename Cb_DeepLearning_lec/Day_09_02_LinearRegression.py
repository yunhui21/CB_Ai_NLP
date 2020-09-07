# Day_09_02_LinearRegression.py
import tensorflow as tf

# 문제 1
# x가 5와 7일 때의 y를 예측하세요


def linear_regression_1():
    x = [1, 2, 3]
    y = [1, 2, 3]

    w = tf.Variable(10.0)
    b = tf.Variable(-5.0)

    # hx = tf.add(tf.multiply(w, x), b)
    hx = w * x + b

    loss_i = tf.square(hx - y)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)
    # train = tf.train.GradientDescentOptimizer(0.1).minimize(tf.reduce_mean(tf.square(w * x + b - y)))

    # print(print('hello'), 'world')

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train)
        print(i, sess.run(loss))

    print('-' * 30)
    print(sess.run(w))
    print(sess.run(b))
    print(sess.run(hx))
    print(sess.run(loss_i))
    print(sess.run(loss))

    print('-' * 30)
    ww, bb = sess.run([w, b])
    print(ww, bb)

    print('5 :', ww * 5 + bb)
    print('7 :', ww * 7 + bb)

    print('5 :', sess.run(w * 5 + b))
    print('7 :', sess.run(w * 7 + b))
    print('* :', sess.run(w * [5, 7] + b))
    print('* :', sess.run(w * x + b))

    sess.close()


def linear_regression_2():
    x = [1, 2, 3]
    y = [1, 2, 3]

    w = tf.Variable(tf.random.uniform([1]))
    b = tf.Variable(tf.random.uniform([1]))

    ph_x = tf.placeholder(tf.float32)

    hx = w * ph_x + b

    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train, feed_dict={ph_x: x})
        print(i, sess.run(loss, {ph_x: x}))

    print('* :', sess.run(hx, {ph_x: x}))
    print('* :', sess.run(hx, {ph_x: [5, 7]}))
    sess.close()


# linear_regression_1()
linear_regression_2()




