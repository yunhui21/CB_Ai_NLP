# Day_33_02_callBacks.py

import tensorflow as tf
import pandas as pd
from sklearn import preprocessing, model_selection
import numpy as np


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


def cars_evaluation_srarse_checkpoints():
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

    # checkpoints = tf.keras.callbacks.ModelCheckpoint('model_cars/cars.h5')
    # checkpoints = tf.keras.callbacks.ModelCheckpoint('model_cars/cars_{epoch:03d}_{val_loss:.4f}.h5',
    #                                                  monitor='val_loss')

    # checkpoints = tf.keras.callbacks.ModelCheckpoint('model_cars/cars_best_{epoch:03d}_{val_loss:.4f}.h5',
    #                                                  monitor='val_loss',
    #                                                  save_best_only=True)

    checkpoints = tf.keras.callbacks.ModelCheckpoint('model_cars/cars_best.h5',
                                                     monitor='val_loss',
                                                     save_best_only=True)

    model.fit(x_train, y_train, epochs=100, batch_size=32,
              validation_data=(x_valid, y_valid), verbose=2,
              callbacks=[checkpoints])
    print('acc :', model.evaluate(x_test, y_test, verbose=0))



def cars_evaluation_srarse_earlystopping():
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

    # data = model_selection.train_test_split(x, y, train_size=train_size+test_size, test_size=test_size)
    # y_train, y_test = data

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(len(set(y)), activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])
    # loss는 높아질때를 acc는 낮아질때를 저장한다.
    # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss')
    # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='acc')
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_acc',
                                                      patience=10) # best값을 갖고 있다는 가정하에

    model.fit(x_train, y_train, epochs=100, batch_size=32,
              validation_data=(x_valid, y_valid), verbose=2,
              callbacks=[early_stopping])
    print('acc :', model.evaluate(x_test, y_test, verbose=0))


# cars_evaluation_srarse_checkpoints()
cars_evaluation_srarse_earlystopping()











