# Day_28_02_Students.py
import numpy as np
from sklearn import model_selection, preprocessing
import tensorflow as tf
import pandas as pd

np.set_printoptions(linewidth=1000)


# 문제
# 학생 성적에 대해 예측하세요 (G1, G2, G3 예측)
# G2는 G1을 포함하고
# G3는 G1과 G2를 포함합니다

def get_data():
    student = pd.read_csv('data/student-mat.csv', delimiter=';')
    student.info()

    enc = preprocessing.LabelEncoder()

    school = enc.fit_transform(student.school)
    sex = enc.fit_transform(student.sex)
    address = enc.fit_transform(student.address)
    famsize = enc.fit_transform(student.famsize)
    Pstatus = enc.fit_transform(student.Pstatus)
    Mjob = enc.fit_transform(student.Mjob)
    Fjob = enc.fit_transform(student.Fjob)
    reason = enc.fit_transform(student.reason)
    guardian = enc.fit_transform(student.guardian)
    schoolsup = enc.fit_transform(student.schoolsup)
    famsup = enc.fit_transform(student.famsup)
    paid = enc.fit_transform(student.paid)
    activities = enc.fit_transform(student.activities)
    nursery = enc.fit_transform(student.nursery)
    higher = enc.fit_transform(student.higher)
    internet = enc.fit_transform(student.internet)
    romantic = enc.fit_transform(student.romantic)

    fits = [school, sex, address, famsize, Pstatus, Mjob,
            Fjob, reason, guardian, schoolsup, famsup,
            paid, activities, nursery, higher, internet, romantic]
    items = [student.age, student.Medu, student.Fedu, student.traveltime, student.studytime,
             student.failures, student.famrel, student.freetime, student.goout,
             student.Dalc, student.Walc, student.health, student.absences,
             student.G1, student.G2, student.G3]

    return np.transpose(fits + items)


def show_difference(preds, labels):
    preds_1 = preds.reshape(-1)
    y_test_1 = labels.reshape(-1)

    diff = preds_1 - y_test_1
    error = np.mean(np.abs(diff))
    print('평균 오차 :', error)


def model_students_regression(x_train, x_test, y_train, y_test):
    n_features = x_train.shape[1]
    n_classes = 1

    w = tf.Variable(tf.random_uniform([n_features, n_classes]))
    b = tf.Variable(tf.random_uniform([n_classes]))

    ph_x = tf.placeholder(tf.float32)

    hx = tf.matmul(ph_x, w) + b

    loss_i = (hx - y_train) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.AdamOptimizer(0.01)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1001):
        sess.run(train, {ph_x: x_train})

        # if i % 100 == 0:
        #     print(i, sess.run(loss, {ph_x: x_train}))

    preds = sess.run(hx, {ph_x: x_test})
    show_difference(preds, y_test)

    sess.close()


data = get_data()
print(data.shape)   # (395, 33)

for i in range(-3, 0):
    x = data[:, :i]
    y = data[:, i].reshape(-1, 1)

    # print(x.shape)      # (395, 30)  (395, 31)  (395, 32)
    # print(y.shape)      # (395, 1)

    items = model_selection.train_test_split(x, y, train_size=0.7)
    x_train, x_test, y_train, y_test = items

    model_students_regression(x_train, x_test, y_train, y_test)

