# Day_17_01_CnnFirstKeras.py
import tensorflow as tf
import numpy as np

# 문제
# 나머지 부분을 코딩하세요

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=[3, 3], strides=[1, 1],
                                 padding='same', activation=tf.keras.activations.relu,
                                 input_shape=[28, 28, 1]))
model.add(tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same'))

model.add(tf.keras.layers.Conv2D(64, [3, 3], [1, 1], 'same', activation=tf.keras.activations.relu))
model.add(tf.keras.layers.MaxPool2D([2, 2], [2, 2], 'same'))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128, tf.keras.activations.relu))
model.add(tf.keras.layers.Dense(10, tf.keras.activations.softmax))

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['acc'])

model.fit(x_train, y_train, epochs=10, batch_size=100, verbose=2, validation_split=0.2)
print(model.evaluate(x_test, y_test, verbose=2))

preds = model.predict(x_test, verbose=2)
preds_arg = np.argmax(preds, axis=1)

print('acc :', np.mean(preds_arg == y_test))

