# Day_25_03_CarEvaluation.py
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import model_selection, preprocessing


# 문제 1
# cars.data 파일을
# x_train, x_test, y_train, y_test로 반환하는 함수를 만드세요

# 문제 2
# 모델을 구축하세요 (앙상블 적용)

def get_data():
    names = ['buying', 'maint', 'doors',
             'persons', 'lug_boot', 'safety', 'class']

    cars = pd.read_csv('data/car.data',
                       header=None,
                       names=names)
    print(cars)

    enc = preprocessing.LabelEncoder()
    # enc.fit_transform(cars.values)    # error

    buying = enc.fit_transform(cars.buying)
    maint = enc.fit_transform(cars.maint)
    doors = enc.fit_transform(cars.doors)
    persons = enc.fit_transform(cars.persons)
    lug_boot = enc.fit_transform(cars.lug_boot)
    safety = enc.fit_transform(cars.safety)
    classes = enc.fit_transform(cars['class'])

    print(buying.shape, buying.dtype)       # (1728,) int32

    data = [buying, maint, doors, persons,
            lug_boot, safety, classes]
    data = np.transpose(data)
    print(data.shape, data.dtype)           # (1728, 7) int32

    x = data[:, :-1]
    y = data[:, -1]         # 반드시 1차원이어야 함

    return model_selection.train_test_split(x, y, train_size=0.7)


def show_accuracy_sparse(preds, labels):
    preds_arg = np.argmax(preds, axis=1)
    # y_arg = np.argmax(labels, axis=1)

    equals = (preds_arg == labels)
    print('acc :', np.mean(equals))


def model_car_evaluation_sparse():
    x_train, x_test, y_train, y_test = get_data()

    n_features = x_train.shape[1]
    n_classes = np.max(y_train) + 1
    w = tf.Variable(tf.random_uniform([n_features, n_classes]))
    b = tf.Variable(tf.random_uniform([n_classes]))

    ph_x = tf.placeholder(tf.float32)

    z = tf.matmul(ph_x, w) + b
    hx = tf.nn.softmax(z)

    loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_train, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train, {ph_x: x_train})
        print(i, sess.run(loss, {ph_x: x_train}))

    # ---------------------------- #

    preds_test = sess.run(hx, {ph_x: x_test})
    show_accuracy_sparse(preds_test, y_test)

    sess.close()


model_car_evaluation_sparse()
