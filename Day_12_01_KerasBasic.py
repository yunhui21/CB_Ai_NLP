# Day_12_01_KerasBasic.py
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection


def logistic_keras():
    x = [[1, 2],
         [2, 1],
         [4, 5],
         [5, 4],
         [8, 9],
         [9, 8]]

    y = [[0], [0], [1], [1], [1], [1]]

    x = np.float32(x)
    y = np.float32(y)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid))

    model.compile(optimizer='sgd',
                  loss = tf.keras.losses.binary_crossentropy,
                  metrics=['acc'])   #

    model.fit(x, y, epochs = 1000, verbose=2)

    print('acc:', model.evaluate((x, y)))

    preds = model.predist(x)


def logistic_indians():

    pima = pd.read_csv('data/pima-indians.csv', )
    # print(pima) #[768 rows x 9 columns]
    #pima.info()

    x = np.float32(pima.values[:, :-1])
    y = np.float32(pima.values[:, -1:])

    print(x.shape, y.shape)     #(768, 8) (768, 1)

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size = 0.7)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid))

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['acc'])  #

    model.fit(x_train, y_train, epochs=10, verbose=2, batch_size=32)

    print('acc:', model.evaluate(x_test, y_test))

    preds = model.predict(x_test)
    # print(preds)
    # print('acc:', np.mean(preds == y_test))#acc: [26.910836442724452, 0.5064935] acc: 0.3116883116883117 같은 값이 되도록 수정해라..
    print('acc:', np.mean((preds > 0.5) == y_test))
logistic_indians()