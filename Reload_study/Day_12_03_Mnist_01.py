# Day_12_03_Mnist_01.py
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection

# mnist = tf.keras.datasets.mnist.load_data()
# print(type(mnist))    # tuple
# print(len(mnist))       # 2
# print(type(mnist[0]), type(mnist[1]))   # <class 'tuple'>, <class 'tuple'>
# print(len(mnist[0]), len(mnist[1]))     # 2, 2


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(x_train.shape, x_test.shape)      # (60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape)      # (60000,) (10000,)

x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)
print(x_train[0])
print(x_test[0])

scalar = preprocessing.MinMaxScaler()
scalar.fit(x_train)

x_train = scalar.transform(x_train)
x_test  = scalar.transform(x_test)

model = tf.keras.Sequential()
model.add(tf.k)