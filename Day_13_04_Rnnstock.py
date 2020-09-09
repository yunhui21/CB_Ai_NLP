# Day_13_04_Rnnstock.py

import tensorflow as tf
import numpy as np
from sklearn import preprocessing, model_selection
import matplotlib.pyplot as plt
import pandas as pd


# 문제
#
def lstm_stock():
    # stock = pd.read_csv('data/stock_daily.csv')
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

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(30, return_sequences=True))
    model.add(tf.keras.layers.LSTM(30))  # 3차원을 받아야 함.
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.mse)

    model.fit(x, y, epochs=10, verbose=2)
    print(model.evaluate(x, y))

    preds = model.predict(x)


lstm_stock()