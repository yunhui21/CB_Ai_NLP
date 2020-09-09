# Day_13_02_SimpleRnn.py
import tensorflow as tf
import numpy as np
# 문제
#

# [0.1, 0.2, 0.3, 0.4] 0.5
# [0.2, 0.3, 0.4, 0.5] 0.6
# [0.3, 0.4, 0.5, 0.6] 0.7

x , y = [], []
for i in np.arange(0, 0.8, 0.1):
    x.append([i, i+0.1, i+0.2, i+0.3])
    y.append(i+0.4)

x = np.float32([x])     # 3차원 변환
y = np.float32([y])

print(x)
print(y)

# a = np.float32([0.1, 0.2, 0.3, 0.4])
# b = np.append(0.1, 0.1).reshape(-1,1)
# print(a + b)

model = tf.keras.Sequential()
model.add(tf.keras.layers.SimpleRNN(7))      # 들어갈 데이터는 hidden_size, 3차원이 들어가야 한다. RNN일때
model.add(tf.keras.layers.Dense(1))

model.compile(optimizer=tf.keras.optimizers.SGD(),
              loss = tf.keras.losses.mse) #regression에서는 metrics를 잘 쓰지 않는다.

model.fit(x, y, epochs=10, verbose= 2)
print(model.evaluate(x, y))

preds = model.predict(x)
print(y)
