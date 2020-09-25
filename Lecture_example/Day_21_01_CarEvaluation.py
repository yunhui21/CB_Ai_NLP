# Day_21_01_CarEvaluation.py
import tensorflow as tf
import pandas as pd
from sklearn import preprocessing, model_selection
import numpy as np


# 문제 1
# car.data 파일을 읽어서 정확도를 계산하세요 (dense)
# 학습, 검증, 검사 데이터로 나누어서 처리하세요

# 문제 2
# car.data 파일을 읽어서 정확도를 계산하세요 (sparse)
# 학습, 검증, 검사 데이터로 나누어서 처리하세요


def get_cars_dense():
    cars = pd.read_csv('data/car.data',
                       header=None,
                       names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'eval'])
    print(cars)

    lb = preprocessing.LabelBinarizer()

    buying = lb.fit_transform(cars.buying)
    maint = lb.fit_transform(cars.maint)
    doors = lb.fit_transform(cars.doors)
    persons = lb.fit_transform(cars.persons)
    lug_boot = lb.fit_transform(cars.lug_boot)
    safety = lb.fit_transform(cars.safety)
    eval = lb.fit_transform(cars['eval'])

    print(buying)

    x = np.hstack([buying, maint, doors, persons, lug_boot, safety])
    y = eval

    print(x.shape, y.shape)     # (1728, 21) (1728, 4)
    return np.float32(x), y


def cars_evaluation_dense():
    x, y = get_cars_dense()

    data = model_selection.train_test_split(x, y, train_size=0.7)
    x_train, x_test, y_train, y_test = data

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(y.shape[1], activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.categorical_crossentropy,        # 'categorical'
                  metrics=['acc'])

    model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=2)
    print('acc :', model.evaluate(x_test, y_test, verbose=0))


def get_cars_sparse():
    cars = pd.read_csv('data/car.data',
                       header=None,
                       names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'eval'])

    lb = preprocessing.LabelEncoder()

    buying = lb.fit_transform(cars.buying)
    maint = lb.fit_transform(cars.maint)
    doors = lb.fit_transform(cars.doors)
    persons = lb.fit_transform(cars.persons)
    lug_boot = lb.fit_transform(cars.lug_boot)
    safety = lb.fit_transform(cars.safety)
    eval = lb.fit_transform(cars['eval'])

    print(buying)

    # x = np.array([buying, maint, doors, persons, lug_boot, safety])       # (6, 1728)
    x = np.transpose([buying, maint, doors, persons, lug_boot, safety])     # (1728, 6)
    y = eval

    print(x.shape, y.shape)     # (1728, 6) (1728,)
    return np.float32(x), y  #, lb.classes_


def cars_evaluation_srarse():
    x, y = get_cars_sparse()

    # 문제
    # x, y를 셔플하세요
    indices = np.arange(len(x))
    np.random.shuffle(indices)

    x = x[indices]
    y = y[indices]

    test_size = int(len(x) * 0.2)
    train_size = len(x) - test_size * 2

    x_train, x_valid, x_test = x[:train_size], x[train_size:train_size+test_size], x[-test_size:]
    y_train, y_valid, y_test = y[:train_size], y[train_size:train_size+test_size], y[-test_size:]
    print(x_train.shape, x_valid.shape, x_test.shape)   # (1038, 6) (345, 6) (345, 6)

    # data = model_selection.train_test_split(x, y, train_size=train_size+test_size, test_size=test_size)
    # y_train, y_test = data

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(len(set(y)), activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])

    model.fit(x_train, y_train, epochs=100, batch_size=32,
              validation_data=[x_valid, y_valid], verbose=2)
    print('acc :', model.evaluate(x_test, y_test, verbose=0))


# cars_evaluation_dense()
cars_evaluation_srarse()













