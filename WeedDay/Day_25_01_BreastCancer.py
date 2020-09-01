# Day_25_01_BreastCancer.py
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import model_selection, preprocessing

# 문제 1
# wdbc 데이터를
# x_train, x_test, y_train, y_test로 반환하는 함수를 만드세요

# 문제 2
# 97.5% 수준의 정확도를 갖는 모델을 구축하세요 (앙상블 적용)


def get_data():
    bc = pd.read_csv('data/wdbc.data',
                     header=None)
    print(bc)
    bc.info()

    enc = preprocessing.LabelEncoder()
    y = enc.fit_transform(bc[1])
    y = y.reshape(-1, 1)
    y = np.float32(y)           # int -> float

    bc.drop([0, 1], axis=1, inplace=True)
    bc.info()

    x = bc.values
    print(x.shape, y.shape)     # (569, 30) (569, 1)

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
    optimizer = tf.train.AdamOptimizer(0.1)
    # optimizer = tf.train.RMSPropOptimizer(0.001)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train, {ph_x: x_train})

        # if i % 10 == 0:
        #     print(i, sess.run(loss, {ph_x: x_train}))

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








