# Day_17_01_CnnFirstKeras.py
import tensorflow as tf
import numpy as np

(x_train, y_train ) , (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(x_train.shape) # (60000, 28, 28)

x_train = x_train.reshape(-1, 28, 28, 1)
x_test  = x_test.reshape(-1, 28, 28, 1)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=[3,3],
                                 strides=[1,1], padding='same',
                                 activation= tf.keras.activations.relu,
                                 input_shape=[28, 28, 1]))
model.add(tf.keras.layers.MaxPool2D(pool_size=[2,2], strides=[2,2], padding='same'))

model.add(tf.keras.layers.Conv2D(64, [3,3],[1,1],'same',activation= tf.keras.activations.relu))
model.add(tf.keras.layers.MaxPool2D([2,2], [2,2], 'same'))

model.add(tf.keras.layers.Flatten())

#Desnse
model.add(tf.keras.layers.Dense(128, tf.keras.activations.relu))
model.add(tf.keras.layers.Dense(10, tf.keras.activations.softmax))

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss = tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['acc'])

model.fit(x_train, y_train, epochs=10, batch_size= 100, verbose=2, validation_split=0.2)
print(model.evaluate(x_test, y_test, verbose=2))
preds = model.predict(x_test, verbose=2)
preds_arg = np.argmax(preds, axis=1)
print('acc:', np.mean(preds_arg == y_test))

# w1 = tf.Variable(tf.random.normal([3, 3, 1, 32])) # weight * height * depth(3, 3, 1) data 1개
# b1 = tf.Variable(tf.zeros([32])) # class, feature 개수

# w2 = tf.Variable(tf.random.normal([3, 3, 32, 64])) # 앞서 입력된 채널값이어야 한다.
# b2 = tf.Variable(tf.zeros([64])) # 크기가 작아서 가능하다.

# w3 = tf.Variable(tf.random.normal([7 * 7 * 64, 128])) # 앞서 입력된 채널값이어야 한다. 1차춴을 다룬다.
# b3 = tf.Variable(tf.zeros([128])) # 크기가 작아서 가능하다.

# w4 = tf.Variable(tf.random.normal([128, 10])) # 앞서 입력된 채널값이어야 한다.
# b4 = tf.Variable(tf.zeros([10])) # 크기가 작아서 가능하다.

# # -------------------------------------------------------- #

# ph_x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
# ph_y = tf.placeholder(tf.int32)

# c1 = tf.nn.conv2d(ph_x, filter = w1, strides = [1,1,1,1], padding = 'SAME') # 가운데는 수직,수평, 양옆은 크기를 맞추는 값. stride
# r1 = tf.nn.relu(c1 + b1)
# p1 = tf.nn.max_pool2d(r1, ksize=[1, 2, 2, 1], strides= [1, 2, 2, 1], padding= 'SAME') # 14*14

# # p1 = tf.nn.max_pool2d(r1, ksize=[1, 2, 2, 1], strides= [1, 2, 2, 1], padding= 'SAME') # 14*14
# # ksize = [1, 3, 3, 1] overlap the flip  겹치는 값을 만든다.

# c2 = tf.nn.conv2d(p1, filter = w2, strides = [1,1,1,1], padding = 'SAME')
# r2 = tf.nn.relu(c2 + b2)
# p2 = tf.nn.max_pool2d(r2, ksize=[1, 2, 2, 1], strides= [1, 2, 2, 1], padding= 'SAME')

# flat = tf.reshape(p2, shape=[-1, 7*7*64]) # 앞쪽에 무슨값이 올지 모른다.

# # w3 = tf.Variable(tf.random.normal([7 * 7 * 64, 128])) # 앞서 입력된 채널값이어야 한다. 1차춴을 다룬다.
# # b3 = tf.Variable(tf.zeros([128])) # 크기가 작아서 가능하다. 하이레벨은 스스로 계산해 내므로 아래로 올수잇다.

# f3 = tf.matmul(flat, w3) + b3
# r3 = tf.nn.relu(f3)

# z = tf.matmul(r3, w4) + b4
# hx = tf.nn.softmax(z)

# loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(logits= z, labels = ph_y)
# loss = tf.reduce_mean(loss_i)

# # optimizer = tf.train.GradientDescentOptimizer(0.1)
# optimizer = tf.train.AdamOptimizer(0.001)
# train = optimizer.minimize(loss)

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())

# epochs = 10
# batch_size = 100
# n_iteration = len(x_train) // batch_size

# for i in range(epochs):
#     total = 0
#     for j in range(n_iteration):
#         n1 = batch_size * j
#         n2 = n1 + batch_size
#
#         xx = x_train[n1:n2]  #shape
#         yy = y_train[n1:n2]
#
#         sess.run(train, {ph_x: xx, ph_y:yy})
#         total = sess.run(loss, {ph_x: xx, ph_y:yy})

#     print(i, total / n_iteration)

# preds = sess.run(z, {ph_x: x_test})
# preds_arg = np.argmax(preds, axis=1)

# print('acc:', np.mean(preds_arg == y_test))
# sess.close()

# print(c1.shape)
# print(p1.shape)
# print(c2.shape)
# print(p2.shape)
# # print(p2.shape, p2.shape[1) * p2.shape[2], * p2.shape[3]
# print(flat.shape)

