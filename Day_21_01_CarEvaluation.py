# Day_21_01_CarEvaluation.py
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import model_selection, preprocessing
import os

# 문제 1
# car_data 파일을 읽어서 정화도를 계산하세요.(dense)
# 학습, 검증, 검사 데이터로 나누어서 처리하세요.

# 문제 2
# car_data 파일을 읽어서 정확도를 계산하세요.(saprse)
# 학습, 검증, 검사 뎅터를 나누어서 처리하세요.

def get_car_dense():
    cars = pd.read_csv('data/car.data', header=None, names=['buying','maint','doors','persons','lug_boot','safety', 'eval'])
    print(cars)
    lb = preprocessing.LabelBinarizer()
    buying   = lb.fit_transform(cars.buying)
    maint    = lb.fit_transform(cars.maint)
    doors    = lb.fit_transform(cars.doors)
    persons  = lb.fit_transform(cars.persons)
    lug_boot = lb.fit_transform(cars.lug_boot)
    safety   = lb.fit_transform(cars.safety)
    eval     = lb.fit_transform(cars['eval'])
    print(buying)

    x = np.hstack([buying, maint, doors, persons, lug_boot, safety])
    y = eval
    print(x.shape, y.shape) # (1728, 21) (1728, 4)6개의 컬럼을 21개로 늘임
    return np.float32(x), y

def cars_evaluation_dense():
    x, y = get_car_dense()
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.7)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(y.shape[1], activation = 'softmax'))

    model.compile(optimizer = tf.keras.optimizers.Adam(0.1),
                  loss = tf.keras.losses.categorical_crossentropy,
                  metrics = ['acc'])
    model.fit(x_train, y_train,  epochs = 100, batch_size= 32, validation_split= 0.1, verbose= 2 ) # sampling개수가 모집단
    print('acc:', model.evaluate(x_test, y_test, verbose=0))

def get_car_sparse():
    cars = pd.read_csv('data/car.data', header=None, names=['buying','maint','doors','persons','lug_boot','safety', 'eval'])

    le = preprocessing.LabelEncoder()
    buying   = le.fit_transform(cars.buying)
    maint    = le.fit_transform(cars.maint)
    doors    = le.fit_transform(cars.doors)
    persons  = le.fit_transform(cars.persons)
    lug_boot = le.fit_transform(cars.lug_boot)
    safety   = le.fit_transform(cars.safety)
    eval     = le.fit_transform(cars['eval'])

    print(buying) #[3 3 3 ... 1 1 1]

    # x = np.array([buying, maint, doors, persons, lug_boot, safety])
    # print(x.shape)
    x = np.transpose([buying, maint, doors, persons, lug_boot, safety])
    # print(x.shape) # (1728, 6)
    y = eval
    # print(x.shape, y.shape) # (1728, 6) (1728,)
    return np.float32(x), y #, le_clsses # (1728, 6) (1728)

def cars_evaluation_sparse():
    x, y = get_car_sparse()

    # 문제
    # x, y를 shuffle해 주세요.

    indices = np.arange(len(x))
    np.random.shuffle(indices)

    x = x[indices]
    y = y[indices]

    test_size  = int(len(x) * 0.2)
    train_size = len(x) - test_size * 2

    x_train, x_valid, x_test = x[:train_size], x[train_size:train_size+test_size], x[-test_size:]
    y_train, y_valid, y_test = y[:train_size], y[train_size:train_size+test_size], y[-test_size:]
    # print(x_train.shape, x_valid.shpae, x_test.shape)
    # x_train, x_test,y_train, y_test = model_selection.train_test_split(x, y, train_size=0.7) #shuffle 데이터를 섞어준다.

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(len(set(y)), activation = 'softmax')) # set으로 개수를 구한다. max사용

    model.compile(optimizer = tf.keras.optimizers.Adam(),
                  loss = tf.keras.losses.sparse_categorical_crossentropy,
                  metrics = ['acc'])
    model.fit(x_train, y_train,  epochs = 100, batch_size= 32,
              validation_data=[x_valid, y_valid],
              verbose= 2 ) # sampling개수가 모집단
    print('acc:', model.evaluate(x_test, y_test, verbose=0))

# get_car_dense()
# cars_evaluation_dense()
# cars_evaluation_sparse()