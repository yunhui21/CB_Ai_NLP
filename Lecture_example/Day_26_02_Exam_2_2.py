# Day_26_02_Exam_2_2.py

# 문제
# fashion-mnist의 test 데이터에 대해 90% 이상의 정확도를 구현하세요
# fashion-mnist의 shape을 변경해서는 안됩니다
# (모델에 전달하는 데이터는 원본 shape을 사용합니다)

import tensorflow as tf

mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

# 맞는 코드는 reshape을 사용할 수 없다
# x_train = x_train.reshape(-1, 28, 28, 1)
# x_test = x_test.reshape(-1, 28, 28, 1)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Input([28, 28]))
model.add(tf.keras.layers.Reshape([28, 28, 1]))     # 추가된 코드

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

model.fit(x_train, y_train, epochs=10, batch_size=100, verbose=2, validation_split=0.2)
print(model.evaluate(x_test, y_test, verbose=2))




