# Day_31_01_Savemodelest.py
import csv
import tensorflow as tf
import numpy as np
from sklearn import model_selection, preprocessing
import pandas as pd


# 문제
# 아발론 데이터에 대해 80% 학습하고 20% 예측하세요.
# evaluate 함수를 사용합니다.

# calss = 29, number of attribute = 8 , number of instance = 4177
'''
Sex		nominal			M, F, and I (infant)
Length		continuous	mm	Longest shell measurement
Diameter	continuous	mm	perpendicular to length
Height		continuous	mm	with meat in shell
Whole weight	continuous	grams	whole abalone
Shucked weight	continuous	grams	weight of meat
Viscera weight	continuous	grams	gut weight (after bleeding)
Shell weight	continuous	grams	after being dried
Rings		integer			+1.5 gives the age in years
'''
def get_abalone():
    abalone = pd.read_csv('data/abalone.data',
                          header=None,
                          names=['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
                                 'Viscera weight', 'Shell weight', 'Rings'])
    # print(abalone)
    # print(abalone.describe())

    x = abalone.values[:, 1:-1]
    y = abalone.Rings.values
    # print(x.shape, y.shape)  # (4177, 7) (4177,)

    sex = preprocessing.LabelBinarizer().fit_transform(abalone.Sex.values)
    # print(sex.shape)  # (4177, 3)

    # 문제
    # 추가하세요.

    # x = np.hstack([x, sex])
    x = np.concatenate([x, sex], axis=1)
    # print(x.shape)  # (4177, 10)

    x = np.float32(x)
    data = model_selection.train_test_split(x, y, train_size=0.8)
    x_train, x_test, y_train, y_test = data

    return x_train, x_test, y_train, y_test

def build_model():


    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(30, activation='softmax')) # 이왕이면 max
    # model.add(tf.keras.layers.Dense(np.max(y)+1, activation='softmax')) # 이왕이면 max

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])

    return  model

def save_load_1():
    x_train, x_test, y_train, y_test = get_abalone()
    model = build_model()
    model.fit(x_train, y_train, epochs=100, verbose=0)
    print('acc :', model.evaluate(x_test, y_test, verbose=0))

    model.save('model_abalone/keras_basic')
    model.save('model_abalone/keras_basic.tf')

save_load_1()