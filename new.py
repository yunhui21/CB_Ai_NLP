# Day_31_01_Exam_5_2.py
import csv
import tensorflow as tf
import numpy as np
import math
import urllib


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w:(w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)


time_step, sunspots = [], []

with open('data/sunspots.csv') as f:
    next(f)     # f.readline()
    for row in csv.reader(f):
        # print(row)
        time_step.append(row[1])
        sunspots.append(row[2])     # str

series = np.float32(sunspots)       # float32
print(series[:3])                   # [ 96.7 104.3 116.7]

# minmax scale : (data - min) / (max - min)
min = np.min(series)
max = np.max(series)

series -= min
series /= max

time = np.array(time_step)      # unused

split_time = 3000               # 학습 데이터 갯수

time_train = time[:split_time]  # unused
time_test = time[split_time:]   # unused
x_train = series[:split_time]
x_test = series[split_time:]

window_size = 30
batch_size = 32
shuffle_buffer = 1000

train_set = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer)
test_set = windowed_dataset(x_test, window_size, batch_size, shuffle_buffer)

for x, y in train_set.take(2):
    print(x.shape, y.shape)

exit(-1)

# 문제
# 앞에 나온 코드를 사용해서 mae 0.12 보다 낮게 예측하세요
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input([None, 1]))
model.add(tf.keras.layers.Conv1D(32, 5, activation='relu'))
model.add(tf.keras.layers.MaxPool1D())
model.add(tf.keras.layers.GRU(32))
model.add(tf.keras.layers.Dense(1))

model.compile(optimizer='rmsprop', loss='mae')
model.fit(train_set, epochs=10,
          verbose=2,
          validation_data=test_set)
