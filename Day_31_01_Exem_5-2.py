# Day_31_01_Exem_5-2.py
import csv
import tensorflow as tf
import numpy as np
import math
import urllib

def windoweed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift= 1, drop_remainder=True)# 한칸식, 자투리는 버리기.
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))# 2차원으로 넘어온 데이터를 하나식 거내기
    ds = ds.shuffle(shuffle_buffer) # 시계열데이터를 순서대로 갖고 와서 셋트로 만들어서 셋트를 섞는다.
    ds = ds.map(lambda w:(w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)# prefetch 미리 데이터를 갖다 놓는다. probider consumer


time_step , sunpsots  =[], []

with open('data/sunspots.csv') as f:
    next(f) # f.reader(f) # header를 건너뜀
    for row in csv.reader(f):
        # print(row)
        time_step.append(row[1])
        sunpsots.append(row[2]) # string

series = np.float32(sunpsots) # float32
print(series[:3]) # [ 96.7 104.3 116.7]

# min, max scaling
# minmax_scale = (data-min)/(max-min)
min = np.min(series)
max = np.max(series)

series -= min
series /= max

time = np.array(time_step) # oneused

split_time = 3000 # 학습 데이터 함수

time_train = time[:split_time]
time_test  = time[split_time:]
x_train = series[:split_time]
x_test  = series[split_time:]

window_size = 30
batch_size = 32
shuffle_buffer = 1000

train_set = windoweed_dataset(x_train, window_size, batch_size, shuffle_buffer)
test_set  = windoweed_dataset(x_test,  window_size, batch_size, shuffle_buffer)

for x, y in train_set.take(2):
    print(x.shape, y.shape)


# 문제
# 코드를 사용해서 mae = 0.12 보다 낮게 예측하세요.

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