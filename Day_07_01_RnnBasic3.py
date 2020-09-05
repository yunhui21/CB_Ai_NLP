# Day_07_01_RnnBasic3.py

import tensorflow as tf
import numpy as np

def show_sequence_loss(targets, logits):
    y = tf.constant(targets)
    z = tf.constant(logits)

    w = tf.ones([1, len(targets[0])])   #바귀지 않는겂 단어에 대한 가중치.

    loss = tf.contrib.seq2seq.sequence_loss(logits= z, targets= y, weights= w)

    sess = tf.Session()
    print(sess.run(loss), targets, logits)
    sess.close()

#       [[  (0, 1)        (0, 1)      (0, 1)]]
pred1 = [[[0.2, 0.7], [0.6, 0.4], [0.1, 0.5]]]
pred2 = [[[0.7, 0.2], [0.4, 0.6], [0.5, 0.1]]]
#       [[  (1, 0)        (1, 0)      (1, 0)]]
# softmax 알고리즘을 이해하라.

show_sequence_loss([[1,1,1]], pred1) # 1이 정답인데 sparse
show_sequence_loss([[0,0,0]], pred2) # softmax의 알고리즘에 의해 0에서 1까지의 거리만 계산할수있다.
# y값이 항상 2차원이 되어야 한다.


# 문제 1
# 아래 코드에서  발생하는 에러를 수정하세요.

# show_sequence_loss([[1,1,1,1]], pred1)
show_sequence_loss([[1,1,1,1]], [[[0.2, 0.7], [0.6, 0.4], [0.1, 0.5], [0.1, 0.5]]])

# 문제2
# 아래 코드에서 발생하는 에를 수정하세요.
show_sequence_loss([[2,2,2]], [[[0.2, 0.7,0.5], [0.6, 0.4,0.3], [0.1, 0.5,0.2]]])

