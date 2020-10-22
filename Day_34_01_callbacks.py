# Day_34_01_callbacks.py

import tensorflow as tf
import pandas as pd
from sklearn import preprocessing, model_selection
import numpy as np

# 문제
# callback중에서 플래터단어가 들어간 것ㅇ르 찻아서
#

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


def cars_evaluation_srarse_plateau():
    x, y = get_cars_sparse()

    indices = np.arange(len(x))
    np.random.shuffle(indices)

    x = x[indices]
    y = y[indices]

    test_size = int(len(x) * 0.2)
    train_size = len(x) - test_size * 2

    x_train, x_valid, x_test = x[:train_size], x[train_size:train_size+test_size], x[-test_size:]
    y_train, y_valid, y_test = y[:train_size], y[train_size:train_size+test_size], y[-test_size:]
    print(x_train.shape, x_valid.shape, x_test.shape)   # (1038, 6) (345, 6) (345, 6)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(len(set(y)), activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.SGD(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])
    # plateau 러니이레이트를 줄여주는 효과
    plateau= tf.keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1)

    model.fit(x_train, y_train, epochs=1000, batch_size=32,
              validation_data=(x_valid, y_valid), verbose=2,
              callbacks=[plateau])
    print('acc :', model.evaluate(x_test, y_test, verbose=0))


cars_evaluation_srarse_plateau()









