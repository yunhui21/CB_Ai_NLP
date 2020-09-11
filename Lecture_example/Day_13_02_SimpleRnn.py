# Day_13_02_SimpleRnn.py
import tensorflow as tf
import numpy as np

# 문제
# 아래와 같은 형태의 데이터를 7개만 만드세요
# [0.1, 0.2, 0.3, 0.4]  0.5
# [0.2, 0.3, 0.4, 0.5]  0.6
# [0.3, 0.4, 0.5, 0.6]  0.7
x, y = [], []
for i in np.arange(0, 0.8, 0.1):
    x.append([i, i+0.1, i+0.2, i+0.3])
    y.append(i+0.4)

x = np.float32([x])
y = np.float32([y])

# a = np.float32([0.1, 0.2, 0.3, 0.4])
# b = np.arange(0, 1, 0.1).reshape(-1, 1)
# print(a + b)

model = tf.keras.Sequential()
model.add(tf.keras.layers.SimpleRNN(7))
model.add(tf.keras.layers.Dense(1))

model.compile(optimizer=tf.keras.optimizers.SGD(),
              loss=tf.keras.losses.mse)

model.fit(x, y, epochs=10, verbose=2)
print(model.evaluate(x, y))
print(model.predict(x))
print(y)
