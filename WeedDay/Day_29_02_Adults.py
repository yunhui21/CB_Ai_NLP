# Day_29_02_Adults.py
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import model_selection, preprocessing


# 문제 1
# 지금까지 배운 모든 걸 동원해서 83% 정확도를 달성해 보세요 (앙상블 제외)
# Day_28_01_BankMarketing.py

def get_data_sparse():
    names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
             'marital_status', 'occupation', 'relationship', 'race', 'sex',
             'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
             'income']
    adult = pd.read_csv('data/adult.data',
                        header=None, names=names)
    print(adult, end='\n\n')
    print(adult.describe(), end='\n\n')

    adult.info()
    print()

    age = adult.age.values
    fnlwgt = adult.fnlwgt.values
    education_num = adult.education_num.values
    capital_gain = adult.capital_gain.values
    capital_loss = adult.capital_loss.values
    hours_per_week = adult.hours_per_week.values

    enc = preprocessing.LabelEncoder()

    workclass = enc.fit_transform(adult.workclass)
    education = enc.fit_transform(adult.education)
    marital_status = enc.fit_transform(adult.marital_status)
    occupation = enc.fit_transform(adult.occupation)
    relationship = enc.fit_transform(adult.relationship)
    race = enc.fit_transform(adult.race)
    sex = enc.fit_transform(adult.sex)
    native_country = enc.fit_transform(adult.native_country)
    income = enc.fit_transform(adult.income)

    x = np.transpose([
        age, fnlwgt, education_num,
        capital_gain, capital_loss, hours_per_week,
        workclass, education, marital_status, occupation,
        relationship, race, sex, native_country
    ])

    # x = preprocessing.minmax_scale(x)
    x = preprocessing.scale(x)

    y = income.reshape(-1, 1)
    y = np.float32(y)

    print(x.shape, y.shape)     # (32561, 14) (32561, 1)

    return model_selection.train_test_split(x, y, train_size=0.7)


def get_data_dense():
    names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
             'marital_status', 'occupation', 'relationship', 'race', 'sex',
             'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
             'income']
    adult = pd.read_csv('data/adult.data',
                        header=None, names=names)

    age = adult.age.values
    fnlwgt = adult.fnlwgt.values
    education_num = adult.education_num.values
    capital_gain = adult.capital_gain.values
    capital_loss = adult.capital_loss.values
    hours_per_week = adult.hours_per_week.values

    enc = preprocessing.LabelBinarizer()

    workclass = enc.fit_transform(adult.workclass)
    education = enc.fit_transform(adult.education)
    marital_status = enc.fit_transform(adult.marital_status)
    occupation = enc.fit_transform(adult.occupation)
    relationship = enc.fit_transform(adult.relationship)
    race = enc.fit_transform(adult.race)
    sex = enc.fit_transform(adult.sex)
    native_country = enc.fit_transform(adult.native_country)
    income = enc.fit_transform(adult.income)

    columns = np.transpose([
        age, fnlwgt, education_num,
        capital_gain, capital_loss, hours_per_week,
    ])
    x = np.hstack([
        columns,
        workclass, education, marital_status, occupation,
        relationship, race, sex, native_country
    ])

    # x = preprocessing.minmax_scale(x)
    x = preprocessing.scale(x)

    y = income.reshape(-1, 1)
    y = np.float32(y)

    print(x.shape, y.shape)     # (32561, 107) (32561, 1)

    return model_selection.train_test_split(x, y, train_size=0.7)


def show_accuracy(preds, labels):
    preds = preds.reshape(-1)

    bools = np.int32(preds > 0.5)
    y_bools = np.int32(labels.reshape(-1))

    # print('acc :', np.mean(bools == y_bools))
    return np.mean(bools == y_bools)


def model_adult(x_train, x_test, y_train, y_test):
    # w = tf.Variable(tf.random_uniform([x_train.shape[1], 1]))

    name = str(np.random.rand())
    w1 = tf.get_variable(name, shape=[x_train.shape[1], 64],
                         initializer=tf.glorot_uniform_initializer)
    b1 = tf.Variable(tf.zeros([64]))

    name = str(np.random.rand())
    w2 = tf.get_variable(name, shape=[64, 32],
                         initializer=tf.glorot_uniform_initializer)
    b2 = tf.Variable(tf.zeros([32]))

    name = str(np.random.rand())
    w3 = tf.get_variable(name, shape=[32, 32],
                         initializer=tf.glorot_uniform_initializer)
    b3 = tf.Variable(tf.zeros([32]))

    name = str(np.random.rand())
    w4 = tf.get_variable(name, shape=[32, 16],
                         initializer=tf.glorot_uniform_initializer)
    b4 = tf.Variable(tf.zeros([16]))

    name = str(np.random.rand())
    w5 = tf.get_variable(name, shape=[16, 1],
                         initializer=tf.glorot_uniform_initializer)
    b5 = tf.Variable(tf.zeros([1]))

    ph_x = tf.placeholder(tf.float32)
    ph_y = tf.placeholder(tf.float32)
    ph_d = tf.placeholder(tf.float32)       # drop-out

    z1 = tf.matmul(ph_x, w1) + b1
    r1 = tf.nn.relu(z1)
    d1 = tf.nn.dropout(r1, keep_prob=ph_d)

    z2 = tf.matmul(d1, w2) + b2
    r2 = tf.nn.relu(z2)
    d2 = tf.nn.dropout(r2, keep_prob=ph_d)

    z3 = tf.matmul(d2, w3) + b3
    r3 = tf.nn.relu(z3)
    d3 = tf.nn.dropout(r3, keep_prob=ph_d)

    z4 = tf.matmul(d3, w4) + b4
    r4 = tf.nn.relu(z4)
    d4 = tf.nn.dropout(r4, keep_prob=ph_d)

    z5 = tf.matmul(d4, w5) + b5

    hx = tf.sigmoid(z5)

    loss_i = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=ph_y, logits=z5)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.AdamOptimizer(0.0001)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    epochs = 100
    batch_size = 32
    n_iteration = len(x_train) // batch_size

    indices = np.arange(len(x_train))

    for i in range(epochs):
        np.random.shuffle(indices)
        c = 0
        for j in range(n_iteration):
            n1 = j * batch_size
            n2 = n1 + batch_size

            part = indices[n1:n2]

            xx = x_train[part]
            yy = y_train[part]

            sess.run(train, {ph_x: xx, ph_y: yy, ph_d: 0.7})
            c += sess.run(loss, {ph_x: xx, ph_y: yy, ph_d: 0.7})

        # print(i, c / n_iteration)

        preds = sess.run(hx, {ph_x: x_test, ph_d: 1.0})
        avg = show_accuracy(preds, y_test)

        print('{:3} : {:7.5f} {:7.5f}'.format(i, c / n_iteration, avg))

    sess.close()
    return preds


x_train, x_test, y_train, y_test = get_data_sparse()
# x_train, x_test, y_train, y_test = get_data_dense()
model_adult(x_train, x_test, y_train, y_test)






