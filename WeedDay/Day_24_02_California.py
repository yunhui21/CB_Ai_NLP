# Day_24_02_California.py
import numpy as np
from sklearn import model_selection, preprocessing, datasets
import tensorflow as tf
import pandas as pd


# 문제 1
# 사이킷런에서 제공하는 캘리포니아 주택가격 데이터를 반환하는 함수를 만드세요
# (x_train, x_test, y_train, y_test)


# 문제 2
# 학습하고 정확도를 구하세요 (미니배치/멀티레이어 사용 금지)

# 문제 3
# 미니배치 알고리즘을 적용하세요

# 문제 4
# 7개로 구성된 앙상블 모델을 만드세요

def show_difference(preds, labels):
    preds_1 = preds.reshape(-1)
    y_test_1 = labels.reshape(-1)

    diff = preds_1 - y_test_1
    print('평균 오차 :', np.mean(np.abs(diff)))
    # print('평균 오차 : {}달러'.format(int(np.mean(np.abs(diff)) * 1000)))


def get_data():
    x, y = datasets.fetch_california_housing(return_X_y=True)
    print(x.shape, y.shape)     # (20640, 8) (20640,)
    print(x.dtype, y.dtype)     # float64 float64

    y = y.reshape(-1, 1)
    x = preprocessing.minmax_scale(x)

    return model_selection.train_test_split(x, y, train_size=0.7)


def model_california(x_train, x_test, y_train, y_test):
    n_features = x_train.shape[1]
    w = tf.Variable(tf.random_uniform([n_features, 1]))

    ph_x = tf.placeholder(tf.float32)
    ph_y = tf.placeholder(tf.float32)

    # (20640, 1) = (20640, 8) @ (8, 1)
    hx = tf.matmul(ph_x, w)

    loss_i = (hx - ph_y) ** 2
    loss = tf.reduce_mean(loss_i)

    # optimizer = tf.train.GradientDescentOptimizer(0.00001)
    optimizer = tf.train.AdamOptimizer(0.01)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    epochs = 10
    batch_size = 100
    n_iteration = len(x_train) // batch_size

    indices = np.arange(len(x_train))

    for i in range(epochs):
        c = 0
        for j in range(n_iteration):
            n1 = j * batch_size
            n2 = n1 + batch_size

            # print(np.array(range(n1, n2)))
            # print(indices[n1:n2])
            # print()

            # xx = x_train[n1:n2]
            # yy = y_train[n1:n2]

            part = indices[n1:n2]
            xx = x_train[part]
            yy = y_train[part]

            sess.run(train, {ph_x: xx, ph_y: yy})
            c += sess.run(loss, {ph_x: xx, ph_y: yy})

        # print(i, c / n_iteration)
        np.random.shuffle(indices)

    preds = sess.run(hx, {ph_x: x_test})
    show_difference(preds, y_test)

    sess.close()
    return preds


x_train, x_test, y_train, y_test = get_data()

results = np.zeros(y_test.shape)
for i in range(7):
    with tf.variable_scope(str(i)):
        preds = model_california(x_train, x_test, y_train, y_test)
        results += preds

print('-' * 30)
results /= 7
show_difference(results, y_test)


