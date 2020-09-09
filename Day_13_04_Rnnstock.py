# Day_13_04_Rnnstock.py

import tensorflow as tf
import numpy as np
from sklearn import preprocessing, model_selection
import matplotlib.pyplot as plt
import pandas as pd


# 문제
#
def lstm_stock():
    stock = np.loadtxt('data/stock_daily.csv', delimiter=',')
    stock = stock[::-1]
    stock = preprocessing.minmax_scale(stock)
    print(stock.shape)          # (732, 5)

    seq_length, n_features = 7, 5
    hidden_size = 9

    rng = [(i, i+seq_length) for i in range(len(stock) - seq_length)]
    print(rng[-1])              # (724, 731)

    x = [stock[s:e] for s, e in rng]
    y = [stock[e][-1] for s, e in rng]
    print(y[-1])
    print(np.array(x).shape, np.array(y).shape)     # (725, 7, 5) (725,)

    x = np.float32(x)
    y = np.reshape(y, [-1, 1])

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.7, shuffle=False)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(hidden_size))  # 3차원을 받아야 함.
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.mse)

    model.fit(x_train, y_train, epochs=10, verbose=2, batch_size=32)
    print('mse:', model.evaluate(x_test, y_test))

    preds = model.predict(x_test)
    preds = preds.reshape(-1)

    idx = range(len(y_test))
    plt.plot(idx, y_test, 'r', label='label')
    plt.plot(idx, preds, 'g', label='prediction')
    plt.legend()
    plt.show()


lstm_stock()