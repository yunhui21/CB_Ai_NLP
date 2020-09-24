# Day_17_02_LeNet.py
import tensorflow as tf
import numpy as np

# 문제
# mnist 데이터셋에 대해 르넷5 버전을 구현하세요


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(6, [5, 5], [1, 1], 'same', activation=tf.keras.activations.relu,
                                 input_shape=[28, 28, 1]))
model.add(tf.keras.layers.MaxPool2D([2, 2], [2, 2], 'same'))

model.add(tf.keras.layers.Conv2D(16, [5, 5], [1, 1], 'valid', activation=tf.keras.activations.relu))
model.add(tf.keras.layers.MaxPool2D([2, 2], [2, 2], 'same'))    # (None, 5, 5, 16)

# model.add(tf.keras.layers.MaxPool2D([2, 2], [2, 2], 'same'))    # (None, 3, 3, 16)
# model.add(tf.keras.layers.MaxPool2D([2, 2], [2, 2], 'valid'))   # (None, 2, 2, 16)

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(120, tf.keras.activations.relu))
model.add(tf.keras.layers.Dense(84, tf.keras.activations.relu))
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


