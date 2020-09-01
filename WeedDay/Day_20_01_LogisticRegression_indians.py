# Day_20_01_LogisticRegression_indians.py


# 문제
# 피마 인디언 당뇨병 데이터로부터
# 70% 데이터로 학습하고 30% 데이터로 정확도를 계산하세요
# Day_19_03_LogisticRegression.py
import tensorflow as tf
import numpy as np
from sklearn import model_selection, preprocessing
import pandas as pd


# def logistic_regression_indians():
#     pima = pd.read_csv('data/pima-indians.csv')
#     # print(pima)
#     # pima.info()
#
#     x = np.float32(pima.values[:, :-1])
#     y = np.float32(pima.values[:, -1:])
#
#     print(x.shape, y.shape)         # (768, 8) (768, 1)
#
#     data = model_selection.train_test_split(x, y, train_size=0.7)
#     x_train, x_test, y_train, y_test = data
#
#     w = tf.Variable(tf.random_uniform([x.shape[1], 1]))     # 8
#     b = tf.Variable(tf.random_uniform([1]))
#
#     ph_x = tf.placeholder(tf.float32)
#
#     # (768, 1) = (768, 8) @ (8, 1)
#     z = tf.matmul(ph_x, w) + b
#     hx = tf.sigmoid(z)
#
#     loss_i = tf.nn.sigmoid_cross_entropy_with_logits(
#         labels=y_train, logits=z)
#     loss = tf.reduce_mean(loss_i)
#
#     optimizer = tf.train.GradientDescentOptimizer(0.0003)
#     train = optimizer.minimize(loss)
#
#     sess = tf.Session()
#     sess.run(tf.global_variables_initializer())
#
#     for i in range(1000):
#         sess.run(train, {ph_x: x_train})
#         print(i, sess.run(loss, {ph_x: x_train}))
#
#     preds = sess.run(hx, {ph_x: x_test})
#     preds = preds.reshape(-1)
#
#     bools = np.int32(preds > 0.5)
#     y_bools = np.int32(y_test.reshape(-1))
#
#     print('acc :', np.mean(bools == y_bools))
#     sess.close()


# 문제
# 학습:검증:검사 데이터로 나누고 (6:2:2)
# 5번에 걸친 학습과 검증 평균 정확도가 65%가 넘도록 만드세요
def logistic_regression_indians_validation():
    pima = pd.read_csv('data/pima-indians.csv')

    x = np.float32(pima.values[:, :-1])
    y = np.float32(pima.values[:, -1:])

    # x = preprocessing.minmax_scale(x)
    x = preprocessing.scale(x)

    print(x.shape, y.shape)         # (768, 8) (768, 1)

    test_size = int(len(x) * 0.2)

    data = model_selection.train_test_split(x, y, test_size=test_size)
    x_total, x_test, y_total, y_test = data

    results = []
    for i in range(1):
        data = model_selection.train_test_split(
            x_total, y_total, test_size=test_size)
        x_train, x_valid, y_train, y_valid = data

        # print(x_train.shape, x_valid.shape, x_test.shape)
        # print(y_train.shape, y_valid.shape, y_test.shape)
        # (462, 8) (153, 8) (153, 8)
        # (462, 1) (153, 1) (153, 1)

        w = tf.Variable(tf.random_uniform([x.shape[1], 1]))     # 8
        b = tf.Variable(tf.random_uniform([1]))

        ph_x = tf.placeholder(tf.float32)

        # (768, 1) = (768, 8) @ (8, 1)
        z = tf.matmul(ph_x, w) + b
        hx = tf.sigmoid(z)

        loss_i = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_train, logits=z)
        loss = tf.reduce_mean(loss_i)

        optimizer = tf.train.GradientDescentOptimizer(0.005)
        train = optimizer.minimize(loss)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for i in range(1000):
            sess.run(train, {ph_x: x_train})

            if i % 10 == 0:
                print(i, sess.run(loss, {ph_x: x_train}))

        preds = sess.run(hx, {ph_x: x_valid})
        preds = preds.reshape(-1)

        bools = np.int32(preds > 0.5)
        y_bools = np.int32(y_valid.reshape(-1))

        acc = np.mean(bools == y_bools)
        print('acc :', acc)
        sess.close()

        results.append(acc)

    # print(results)
    print(np.mean(results))


# logistic_regression_indians()
logistic_regression_indians_validation()





