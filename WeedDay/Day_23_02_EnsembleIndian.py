# Day_23_02_EnsembleIndian.py
import tensorflow as tf
import numpy as np
from sklearn import model_selection, preprocessing
import pandas as pd


# 문제
# 피마 인디언 당뇨병 데이터를 7대 3으로 나누어서
# 75%의 정확도를 구현하세요
# (앙상블을 포함해서 지금까지 배운 모든 것을 활용합니다)
def get_data():
    pima = pd.read_csv('data/pima-indians.csv')

    x = np.float32(pima.values[:, :-1])
    y = np.float32(pima.values[:, -1:])

    # x = preprocessing.minmax_scale(x)
    x = preprocessing.scale(x)

    return model_selection.train_test_split(x, y, train_size=0.7)


def show_accuracy(preds, labels):
    preds = preds.reshape(-1)

    bools = np.int32(preds > 0.5)
    y_bools = np.int32(labels.reshape(-1))

    print('acc :', np.mean(bools == y_bools))


def logistic_regression_indians(x_train, x_test, y_train, y_test):
    name = 'w' + str(np.random.rand(1)[0])
    w = tf.get_variable(name, shape=[x_train.shape[1], 1],
                        initializer=tf.glorot_uniform_initializer)
    b = tf.Variable(tf.zeros([1]))

    ph_x = tf.placeholder(tf.float32)

    z = tf.matmul(ph_x, w) + b
    hx = tf.sigmoid(z)

    loss_i = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=y_train, logits=z)
    loss = tf.reduce_mean(loss_i)

    # optimizer = tf.train.GradientDescentOptimizer(0.0003)
    # optimizer = tf.train.AdamOptimizer(0.01)
    optimizer = tf.train.RMSPropOptimizer(0.001)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train, {ph_x: x_train})
        # print(i, sess.run(loss, {ph_x: x_train}))

    preds = sess.run(hx, {ph_x: x_test})
    show_accuracy(preds, y_test)

    sess.close()

    return preds


x_train, x_test, y_train, y_test = get_data()

results = np.zeros(y_test.shape)
for i in range(7):
    preds = logistic_regression_indians(x_train, x_test, y_train, y_test)
    results += preds

print('-' * 30)
show_accuracy(results / 7, y_test)
