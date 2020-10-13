# Day_10_02_JenaClimate.py
import tensorflow as tf
import numpy as np
from sklearn import preprocessing, model_selection
import pandas as pd
import matplotlib.pyplot as plt

# 문제 1
# Jena 데이터를 온도만 추출해서 그래프로 그리세요.

# 문제 2
# jena의 온도 데이터를 리니어 리그레션으로 예측하세요.
# 예측 결과와 정답을 같은 그래프에 그려 주세요.

def show_temperature():
    jena = pd.read_csv('data/jena_climate_2009_2016.csv', index_col=0)
    print(jena) # [420551 rows x 14 columns]
    print(jena.columns.values)

    '''
    ['p (mbar)' 'T (degC)' 'Tpot (K)' 'Tdew (degC)' 'rh (%)' 'VPmax (mbar)'
    'VPact (mbar)' 'VPdef (mbar)' 'sh (g/kg)' 'H2OC (mmol/mol)'
    'rho (g/m**3)' 'wv (m/s)' 'max. wv (m/s)' 'wd (deg)']
    '''

    # degc = jena['T (degC)'].values
    # idx = range(len(degc))
    # plt.plot(idx, degc)

    degc = jena['T (degC)']
    degc.plot(subplots=True)
    plt.show()

def jena_regression():
    jena = pd.read_csv('data/jena_climate_2009_2016.csv', index_col=0)
    degc = jena['T (degC)'].values

    x = degc[:-1]
    y = degc[1:]

    w = tf.Variable(5.0)
    b = tf.Variable(-3.0)

    hx = w * x + b

    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.001)
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(i, sess.run(loss))

    preds = sess.run(hx)
    idx = range(len(y))

    # plt.plot(idx, y, 'r')
    # plt.plot(idx, preds, 'g')
    # plt.show()
    print('mse:', np.mean(np.abs(preds-y)))
    # degc = jena['T (degC)'].values
    # idx = range(len(degc))
    # plt.plot(idx, degc)
    # pht.show()

    # degc = jena['T (degC)']
    # degc.plot(subplots=True)
    # plt.show()

# show_temperature()
jena_regression()

