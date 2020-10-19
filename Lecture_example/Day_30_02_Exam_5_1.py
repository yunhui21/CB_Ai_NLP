# Day_30_02_Exam_5_1.py
import csv
import tensorflow as tf
import numpy as np
import urllib
import pandas as pd
from sklearn import model_selection, preprocessing

# 문제
# 30개의 데이터를 이용해서 다음 결과를 예측하는 모델을 만드세요
# 학습에 3000개, 검사에 나머지 사용
# 통과 기준: mae 0.12

# url = 'https://storage.googleapis.com/download.tensorflow.org/data/Sunspots.csv'
# urllib.request.urlretrieve(url, 'data/sunspots.csv')

sunspots = pd.read_csv('data/sunspots.csv', index_col=0)
# print(sunspots)

series = sunspots['Monthly Mean Total Sunspot Number'].values
# print(series)

# print(series[:5])
series = preprocessing.minmax_scale(series)
# print(series[:5])

seq_length, n_features = 30, 1
hidden_size = 9

rng = [(i, i + seq_length) for i in range(len(series) - seq_length)]

x = np.float32([series[s:e] for s, e in rng])
y = np.float32([series[e] for s, e in rng])
print(x.shape, y.shape)

# x = x[:, :, np.newaxis]
x = x.reshape(x.shape[0], x.shape[1], 1)

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=3000, shuffle=False)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Input([None, n_features]))
model.add(tf.keras.layers.Conv1D(32, 5, activation='relu'))
model.add(tf.keras.layers.MaxPool1D())
model.add(tf.keras.layers.GRU(32))
model.add(tf.keras.layers.Dense(1))

model.compile(optimizer='rmsprop', loss='mae')
model.fit(x_train, y_train, epochs=10, batch_size=32,
          verbose=2,
          validation_data=(x_test, y_test))


