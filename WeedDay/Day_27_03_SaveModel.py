# Day_27_03_SaveModel.py
import tensorflow as tf


def linear_regression_save():
    x = [1, 2, 3]
    y = [1, 2, 3]

    w = tf.Variable(5.0)
    b = tf.Variable(-3.0)

    hx = w * x + b

    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    for i in range(10):
        sess.run(train)
        print(sess.run(loss))
        saver.save(sess, 'model/series', global_step=i)

    # saver.save(sess, 'model/linear')

    print('5 :', sess.run(w * 5 + b))   # 5 : 8.750338
    print('7 :', sess.run(w * 7 + b))   # 7 : 13.501097

    sess.close()


def linear_regression_restore():
    # x = [1, 2, 3]
    # y = [1, 2, 3]

    w = tf.Variable(5.0)
    b = tf.Variable(-3.0)

    # hx = w * x + b
    #
    # loss_i = (hx - y) ** 2
    # loss = tf.reduce_mean(loss_i)
    #
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    # train = optimizer.minimize(loss=loss)

    latest = tf.train.latest_checkpoint('model')
    print(latest)

    sess = tf.Session()
    # sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    saver.restore(sess, latest)

    print('5 :', sess.run(w * 5 + b))   # 5 : 8.750338
    print('7 :', sess.run(w * 7 + b))   # 7 : 13.501097

    sess.close()


linear_regression_save()
# linear_regression_restore()

