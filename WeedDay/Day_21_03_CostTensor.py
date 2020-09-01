# Day_21_03_CostTensor.py
import tensorflow as tf         # 1.14.0
import numpy as np
import matplotlib.pyplot as plt


# 문제
# 파이썬으로 만들었던 cost 그래프를
# 텐서플로 버전으로 변환하세요
def cost_tensor_1():
    x = [1, 2, 3]
    y = [1, 2, 3]

    ph_w = tf.placeholder(tf.float32)
    hx = ph_w * x

    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(-30, 50):
        w = i / 10
        c = sess.run(loss, {ph_w: w})
        print(w, c)
        plt.plot(w, c, 'ro')

    plt.show()

    sess.close()


# 문제
# 반복문을 없애보세요
def cost_tensor_2():
    x = [1, 2, 3]
    y = [1, 2, 3]

    w = np.arange(-3, 5, 0.1).reshape(-1, 1)
    hx = w * x

    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i, axis=1)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    c = sess.run(loss)
    print(c.shape)
    plt.plot(w, c, 'ro')
    plt.show()

    sess.close()


def cost_numpy():
    x = [1, 2, 3]
    y = [1, 2, 3]

    w = np.arange(-3, 5, 0.1).reshape(-1, 1)
    hx = w * x

    loss_i = (hx - y) ** 2
    c = np.mean(loss_i, axis=1)

    plt.plot(w, c, 'ro')
    plt.show()


# cost_tensor_1()
# cost_tensor_2()
cost_numpy()
