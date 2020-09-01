# Day_06_01_RnnBasic.py

import tensorflow as tf
import numpy as np


# char rnn
# word rnn
# 학교에 (간다, 갈까?, 같이 갈까?, 행사가)
# 1... 2... 3...
# 학교에 가서 (밥을, 친구를, 할일없이)
# x : tenso
# y : ensor

# 문제 .1
# 'tensor' 문자열로 x, y 데이터로 만드세요.
# w = 'tensor' # 5개의 글자가 tenso x_data, y_data r 1개
# tensor : class(중복되지 않는 레이블) 6개 - 예측해야하는 개수
# idx2char = ['t', 'e', 'n', 's', 'o', 'r'] : 고민
# tensor 순서대로 정렬해본다. ensorst : 012345, tensor : 501423
# 20_02_sofmax참고

x = [[0, 0, 0, 0, 0, 1],    # e
     [1, 0, 0, 0, 0, 0],    # n
     [0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 0],
     [0, 0, 1, 0, 0, 0]]
y = [[1, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 0],
     [0, 0, 1, 0, 0, 0],
     [0, 0, 0, 1, 0, 0]]

# 문제 .2
x = [[0, 0, 0, 0, 0, 1],  # e
     [1, 0, 0, 0, 0, 0],  # n
     [0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 0],
     [0, 0, 1, 0, 0, 0]]
y = [[1, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 0],
     [0, 0, 1, 0, 0, 0],
     [0, 0, 0, 1, 0, 0]]

x = np.float32(x)

w = tf.Variable(tf.random.uniform([6,6]))
b = tf.Variable(tf.random.uniform([6])) # bias
# (5, 6) = (5, 6) @ (6, 6)

z = tf.matmul(x, w)
hx = tf.nn.softmax(z) # y

loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=z)
loss = tf.reduce_mean(loss_i)

optimizer =tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(100):
    sess.run(train)
    c = sess.run(loss)
    preds = sess.run(hx)
    preds_arg = np.argmax(preds, axis = 1)
    print(i, c, preds_arg)

sess.close()
# 99 0.7226122 [0 1 4 2 3]




