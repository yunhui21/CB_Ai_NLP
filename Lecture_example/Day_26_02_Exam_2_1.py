# Day_26_02_Exam_2_1.py

# 문제
# mnist의 test 데이터에 대해 90% 이상의 정확도를 구현하세요
# mnist의 shape을 변경해서는 안됩니다
# (모델에 전달하는 데이터는 원본 shape을 사용합니다)

import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.dtype)
# x_train = np.float32(x_train)

x_train = x_train / 255
x_test = x_test / 255

# 틀린 코드. 맞는 코드는 fashion_mnist를 보세요
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

model = tf.keras.Sequential()
# model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=[3, 3], strides=[1, 1],
#                                  padding='same', activation=tf.keras.activations.relu,
#                                  input_shape=[28, 28, 1]))
# model.add(tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same'))

model.add(tf.keras.layers.Input([28, 28, 1]))
model.add(tf.keras.layers.Conv2D(32, [3, 3], [1, 1], 'same', activation=tf.keras.activations.relu))
model.add(tf.keras.layers.MaxPool2D([2, 2], [2, 2], 'same'))

model.add(tf.keras.layers.Conv2D(64, [3, 3], [1, 1], 'same', activation=tf.keras.activations.relu))
model.add(tf.keras.layers.MaxPool2D([2, 2], [2, 2], 'same'))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128, tf.keras.activations.relu))
model.add(tf.keras.layers.Dense(10, tf.keras.activations.softmax))

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['acc'])

model.fit(x_train, y_train, epochs=1, batch_size=100, verbose=2, validation_split=0.2)
print(model.evaluate(x_test, y_test, verbose=2))





