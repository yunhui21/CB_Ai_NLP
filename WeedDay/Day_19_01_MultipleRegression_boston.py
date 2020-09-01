# Day_19_01_MultipleRegression_boston.py
import tensorflow as tf
import pandas as pd
from sklearn import model_selection
import numpy as np


# 문제 1
# 보스턴 주택가격 데이터를 pandas로 읽어오세요
boston = pd.read_excel('data/BostonHousing.xls')
# print(boston)

# 문제 2
# 보스턴 데이터로부터 x, y 데이터를 구축하세요 (drop 함수)
# y = boston.MEDV.values.reshape(-1, 1)
# x = boston.drop(['MEDV'], axis=1).values

values = boston.values

x = values[:, :-1]
y = values[:, -1:]

# print(y.shape, x.shape) # (506, 1) (506, 13)

# 문제 3
# 멀티플 리그레션을 이용해서 학습하세요

# 문제 4
# 마지막 데이터를 제외한 데이터로 학습하고, 마지막 데이터 1개에 대해 예측하세요
# x_train, x_test = x[:-1], x[-1:]
# y_train, y_test = y[:-1], y[-1:]

# 문제 5
# 70%로 학습하고 30%에 대해 예측하세요
# 그리고, 오차 평균을 구하세요
data = model_selection.train_test_split(x, y, train_size=0.7)
x_train, x_test, y_train, y_test = data

w = tf.Variable(tf.random_uniform([x.shape[1], 1]))
b = tf.Variable(tf.random_uniform([1]))
ph_x = tf.placeholder(tf.float32)

# (506, 1) = (506, 13) @ (13, 1)
hx = tf.matmul(ph_x, w) + b

loss_i = (hx - y_train) ** 2
loss = tf.reduce_mean(loss_i)

optimizer = tf.train.GradientDescentOptimizer(0.000001)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    sess.run(train, {ph_x: x_train})
    print(i, sess.run(loss, {ph_x: x_train}))

preds = sess.run(hx, {ph_x: x_test})
print(preds.shape)
print(y_test.shape)

preds_1 = preds.reshape(-1)
y_test_1 = y_test.reshape(-1)

diff = preds_1 - y_test_1
print(diff[:5])
print('평균 오차 :', np.mean(np.abs(diff)))
print('평균 오차 : {}달러'.format(int(np.mean(np.abs(diff)) * 1000)))

sess.close()
