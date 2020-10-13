# Day_26_02exem2-2.py
# 문제
# mnist test 데이터에 대해 sgd  정확도를 구현하세요.
# mnist의 shape을 변경해서는 안됩니다.


import tensorflow as tf
from sklearn import preprocessing
import numpy as np



# Day_17_01_CnnFirstKeras.py
import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train ) , (x_test, y_test) = mnist.load_data()
# print(x_train.shape) # (60000, 28, 28)
x_train = x_train/255
x_test = x_test/255

x_train = x_train.reshape(-1, 28, 28, 1)
x_test  = x_test.reshape(-1, 28, 28, 1)

model = tf.keras.Sequential()
# model.add(tf.keras.layers.Input([28,28]))
model.add(tf.keras.layers.Reshape([28,28, 1]))

model.add(tf.keras.layers.Conv2D(64, [3,3],[1,1],'same',activation= tf.keras.activations.relu))
model.add(tf.keras.layers.MaxPool2D([2,2], [2,2], 'same'))

model.add(tf.keras.layers.Conv2D(64, [3,3],[1,1],'same',activation= tf.keras.activations.relu))
model.add(tf.keras.layers.MaxPool2D([2,2], [2,2], 'same'))

model.add(tf.keras.layers.Flatten())

#Desnse
model.add(tf.keras.layers.Dense(128, tf.keras.activations.relu))
model.add(tf.keras.layers.Dense(10, tf.keras.activations.softmax))


model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss = tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['acc'])

model.fit(x_train, y_train, epochs=10, batch_size= 100, verbose=2, validation_split=0.2)
print(model.evaluate(x_test, y_test, verbose=2))

