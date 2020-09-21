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
    return x, y

def cars_evaluation_dense():
    # get_car_dense()

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.7)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(32, tf.keras.activations.softmax))

    model.compile(optimizer = tf.keras.optimizers.Adam(0.1),
                  loss = tf.keras.losses.categorical_crossentropy,
                  metrics= ['acc'])
    model.fit(x_train, y_train,  epochs = 100, verbose= 2 )
    print('acc:',model.fit)
# get_car_dense()
cars_evaluation_dense()
