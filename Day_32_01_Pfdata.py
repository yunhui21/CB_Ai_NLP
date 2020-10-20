# Day_32_01_Pfdata.py

import csv
import tensorflow as tf
import numpy as np
from sklearn import model_selection, preprocessing
import pandas as pd

# 문제
# Day_31_02_savemodelist.py 파일의 abalan 데이터를
# tf.keras.datasets을 사용해서 재구성하세요.


def get_abalone_by_tfdata():
    abalone = pd.read_csv('data/abalone.data',
                          header=None,
                          names=['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
                                 'Viscera weight', 'Shell weight', 'Rings'])

    # x, y는 같은 행에 있어야 한다. 같이 섞여야 한다.
    data = abalone.values[:, 1:]
    sex = preprocessing.LabelBinarizer().fit_transform(abalone.Sex.values)

    data = np.hstack([data, sex])
    data = np.float32(data)

    train_size = int(len(data) * 0.8)
    data_train = data[:train_size]
    data_test  = data[train_size:]

    ds_train = tf.data.Dataset.from_tensor_slices(data_train)
    ds_train = ds_train.batch(32, drop_remainder=True)

    # for item in ds_train.take(2):
    #     print(item.shape) # (32, 11)

    ds_train = ds_train.shuffle(buffer_size= 10000) # 32개로 묶여 있는 세트가 shuffle
    ds_train = ds_train.map(lambda chunk: (chunk[:, :-1], chunk[:, -1]))

    # for xx, yy in ds_train.take(2):
    #     print(xx.shape, yy.shape)

    ds_test = tf.data.Dataset.from_tensor_slices(data_test)
    ds_test = ds_test.batch(32, drop_remainder=True)

    # for item in ds_train.take(2):
    #     print(item.shape) # (32, 11)

    ds_test = ds_test.shuffle(buffer_size=10000)  # 32개로 묶여 있는 세트가 shuffle
    ds_test = ds_test.map(lambda chunk: (chunk[:, :-1], chunk[:, -1]))

    return ds_train, ds_test

def build_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(30, activation='relu'))
    model.add(tf.keras.layers.Dense(30, activation='softmax'))  # 이왕이면 max
    # model.add(tf.keras.layers.Dense(np.max(y)+1, activation='softmax')) # 이왕이면 max

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])

    return model


def model_abalon():

    # x_train , y_train을 묶는것은 안됨.
    # tf.data.Dataset.from_tensor_slices([x_train, y_train]) # error
    # tf.data.Dataset.from_tensor_slices(zip([x_train, y_train])) # error
    # tf.data.Dataset.from_tensor_slices(list(zip([x_train, y_train]))) # x_train 여러개를 y_train은 한개
    # tf.data.Dataset.from_tensor_slices(np.array(zip([x_train, y_train])))

    # n = [[3, 2, 4],
    #      [2, 1, 3],
    #      [4, 6, 5]]
    # tf.data.Dataset.from_tensor_slices(n)

    ds_train, ds_test = get_abalone_by_tfdata()

    model = build_model()

    # model.fit(ds_train, epochs=100, verbose=2, validation_data=ds_test)
    model.fit(ds_train.repeat(),steps_per_epoch=5, epochs=10, verbose=2,
              validation_data= ds_test.repeat(),
              validation_steps=1)

    print('acc :', model.evaluate(ds_test, verbose=0))

# get_abalone_by_tfdata()
model_abalon()