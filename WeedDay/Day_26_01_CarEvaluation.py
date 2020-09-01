# Day_26_01_CarEvaluation.py
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import model_selection, preprocessing


# 문제 1
# 25-3 파일을 복사해서 미니배치 방식으로 수정하세요

# 문제 2
# sparse 데이터를 사용하지 원핫 벡터 형태의 데이터를 사용해서
# 이전과 동일한 코드를 만드세요
# LabelEncoder ==> LabelBinarizer


def get_data():
    names = ['buying', 'maint', 'doors',
             'persons', 'lug_boot', 'safety', 'class']

    cars = pd.read_csv('data/car.data',
                       header=None,
                       names=names)
    print(cars)

    enc = preprocessing.LabelBinarizer()
    # enc.fit_transform(cars.values)    # error

    buying = enc.fit_transform(cars.buying)
    maint = enc.fit_transform(cars.maint)
    doors = enc.fit_transform(cars.doors)
    persons = enc.fit_transform(cars.persons)
    lug_boot = enc.fit_transform(cars.lug_boot)
    safety = enc.fit_transform(cars.safety)
    classes = enc.fit_transform(cars['class'])

    print(buying.shape, buying.dtype)       # (1728, 4) int32

    data = [buying, maint, doors, persons, lug_boot, safety]
    data = np.hstack(data)
    print(data.shape, data.dtype)           # (1728, 21) int32

    x = data[:, :-1]
    y = classes
    print(y.shape)                          # (1728, 4)

    return model_selection.train_test_split(x, y, train_size=0.7)


def show_accuracy_dense(preds, labels):
    preds_arg = np.argmax(preds, axis=1)
    y_arg = np.argmax(labels, axis=1)

    equals = (preds_arg == y_arg)
    print('acc :', np.mean(equals))


def model_car_evaluation_dense():
    x_train, x_test, y_train, y_test = get_data()

    n_features = x_train.shape[1]
    n_classes = y_train.shape[1]
    w = tf.Variable(tf.random_uniform([n_features, n_classes]))
    b = tf.Variable(tf.random_uniform([n_classes]))

    ph_x = tf.placeholder(tf.float32)
    ph_y = tf.placeholder(tf.int32)

    z = tf.matmul(ph_x, w) + b
    hx = tf.nn.softmax(z)

    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(labels=ph_y, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.AdamOptimizer(0.01)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    epochs = 10
    batch_size = 32
    n_iteration = len(x_train) // batch_size

    indices = np.arange(len(x_train))

    for i in range(epochs):
        c = 0
        for j in range(n_iteration):
            n1 = j * batch_size
            n2 = n1 + batch_size

            xx = x_train[n1:n2]
            yy = y_train[n1:n2]

            sess.run(train, {ph_x: xx, ph_y: yy})
            c += sess.run(loss, {ph_x: xx, ph_y: yy})

        print(i, c / n_iteration)
        np.random.shuffle(indices)

    # ---------------------------- #

    preds_test = sess.run(hx, {ph_x: x_test})
    show_accuracy_dense(preds_test, y_test)

    sess.close()


model_car_evaluation_dense()

# http://210.125.150.125/
