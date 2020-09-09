# Day_13_01_DeepReview.py

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, datasets

# 문제
# 캘리포니아 주책 가격에 대해서
# dataset 6:2:2로 나눠서 결과를 예측하세요.

def california_housing():
    x, y = datasets.fetch_california_housing(return_X_y=True)
    print(x.shape)      # (20640, 8)
    print(y.shape)      # (20640,)
    print(type(x))      # <class 'numpy.ndarray'>

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.8)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid))

    model.compile(optimizer=tf.keras.optimizers.SGD(),
                  loss = tf.keras.losses.mse,
                  metrics =['acc'])

    model.fit(x_train, y_train, epochs=1, verbose= 2)
    print(model.evaluate(x_test, y_test))

    preds = model.predict(x_test)
    print(preds.shape, y_test.shape)    # (4128, 1) (4128,)

    print(np.mean(np.abs(preds.reshape(-1) - y_test)))

california_housing()