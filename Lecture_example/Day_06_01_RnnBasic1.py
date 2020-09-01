# Day_06_01_RnnBasic1.py
import tensorflow as tf
import numpy as np

# char rnn
# word rnn
# 학교에 (간다, 갈까?, 같이 갈래?, 행사가)
#  1 ..  2..  3..
# 학교에 가서 (밥을, 친구를, 할일없이)
# x : tenso
# y : ensor

# 문제 1
# 'tensor' 문자열로 x, y 데이터를 만드세요 (벡터화 필수, 원핫 벡터 생성)
# tensor : 501423
# enorst : 012345

# 문제 2
# 앞에서 생성한 데이터에 대해 결과를 예측하세요 (softmax)

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

x = np.float32(x)

w = tf.Variable(tf.random.uniform([6, 6]))
b = tf.Variable(tf.random.uniform([6]))

# (5, 6) = (5, 6) @ (6, 6)
z = tf.matmul(x, w) + b
hx = tf.nn.softmax(z)

loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=z)
loss = tf.reduce_mean(loss_i)

optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(100):
    sess.run(train)
    c = sess.run(loss)
    preds = sess.run(hx)
    preds_arg = np.argmax(preds, axis=1)
    print(i, c, preds_arg)

sess.close()




