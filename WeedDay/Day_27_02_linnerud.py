# Day_27_02_linnerud.py
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import model_selection, preprocessing, datasets


# 문제 1
# linnerud 데이터에 대해 오차를 구하세요 (개별적으로)

# 문제 2
# linnerud 데이터에 대해 오차를 구하세요 (전체적으로)


def basic():
    rud = datasets.load_linnerud()
    print(rud.keys())
    # ['data', 'feature_names', 'target',
    # 'target_names', 'frame', 'DESCR', 'data_filename',
    # 'target_filename']

    print(rud['feature_names'])     # ['Chins', 'Situps', 'Jumps']
    print(rud['target_names'])      # ['Weight', 'Waist', 'Pulse']

    print(rud['data'].shape)        # (20, 3)


def show_difference(preds, labels):
    preds_1 = preds.reshape(-1)
    y_test_1 = labels.reshape(-1)

    diff = preds_1 - y_test_1
    error = np.mean(np.abs(diff))
    print('평균 오차 :', error)


def show_difference_all(preds, labels):
    diff = preds - labels
    error = np.mean(np.abs(diff), axis=0)
    print('평균 오차 :', error)


def model_linnerud_by_one(x, y):
    n_features = x.shape[1]
    n_classes = 1

    w = tf.Variable(tf.random_uniform([n_features, n_classes]))
    b = tf.Variable(tf.random_uniform([n_classes]))

    hx = tf.matmul(x, w) + b

    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.AdamOptimizer(0.01)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train)
        # print(i, sess.run(loss))

    preds = sess.run(hx)
    show_difference(preds, y)
    sess.close()


def model_linnerud_by_all(x, y):
    w = tf.Variable(tf.random_uniform([3, 3]))
    b = tf.Variable(tf.random_uniform([3]))

    # (20, 3) = (20, 3) @ (3, 3)
    hx = tf.matmul(x, w) + b

    # (20, 3) = (20, 3) - (20, 3)
    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i, axis=0)

    optimizer = tf.train.AdamOptimizer(0.01)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(i, sess.run(loss))

    preds = sess.run(hx)
    show_difference_all(preds, y)
    sess.close()


def model_linnerud_by_all_2(x, y):
    w = tf.Variable(tf.random_uniform([3, 1]))
    b = tf.Variable(tf.random_uniform([1]))

    # (20, 1) = (20, 3) @ (3, 1)
    hx = tf.matmul(x, w) + b

    # (20, 3) = (20, 1) - (20, 3)
    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i, axis=0)

    optimizer = tf.train.AdamOptimizer(0.01)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(i, sess.run(loss))

    preds = sess.run(hx)
    show_difference_all(preds, y)
    sess.close()


# basic()

x, y = datasets.load_linnerud(return_X_y=True)
x = np.float32(x)
y = np.float32(y)
print(x.shape, y.shape)     # (20, 3) (20, 3)

# 문제
# model_linnerud_by_one 함수를 호출해 보세요 (에러나지 않게)

# for i in range(3):
#     # model_linnerud_by_one(x, y[:, 0].reshape(-1, 1))
#     model_linnerud_by_one(x, y[:, i:i+1])


# model_linnerud_by_all(x, y)
model_linnerud_by_all_2(x, y)
