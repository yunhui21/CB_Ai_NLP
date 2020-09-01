# Day_06_02_RnnBasic2.py
import tensorflow as tf
import numpy as np

x = [[0, 0, 0, 0, 0, 1],    # t
     [1, 0, 0, 0, 0, 0],    # e
     [0, 1, 0, 0, 0, 0],    # n
     [0, 0, 0, 0, 1, 0],    # s
     [0, 0, 1, 0, 0, 0],]   # o
y = [[1, 0, 0, 0, 0, 0],    # e
     [0, 1, 0, 0, 0, 0],    # n
     [0, 0, 0, 0, 1, 0],    # s
     [0, 0, 1, 0, 0, 0],    # o
     [0, 0, 0, 1, 0, 0],]   # r

x = np.float32([x])         # 2차원 -> 3차원

hidden_size = 6
cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
outputs, _states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

print(outputs.shape)    # (1, 5, 6)
z = outputs[0]

loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=z)
loss = tf.reduce_mean(loss_i)

optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(100):
    sess.run(train)
    c = sess.run(loss)
    preds = sess.run(z)
    preds_arg = np.argmax(preds, axis=1)
    print(i, c, preds_arg)

sess.close()




