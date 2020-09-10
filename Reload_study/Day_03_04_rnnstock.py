# Day_03_04_rnnstock.py

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection
import matplotlib as plt


def lstm_stock():

    stock = np.loadtxt('data/stock_daily.csv', delimiter=',')
    stock = stock[::-1]
    stock = preprocessing.minmax_scale(stock)
    print(stock.shape)

    seq_length, n_features = 7, 5
    hidden_size = 9

    rng = [(i, i+seq_length) for i in range(len(stock)-seq_length)]

    x = [stock[s:e] for s,e in rng]
    y = [stock[e][-1] for s,e in rng]
    print(y[-1])
lstm_stock()