# Day_09_01_BasicRnn8.py
import tensorflow as tf
import numpy as np
from sklearn import preprocessing

# 문제
# 굉장히 긴 문자열을 학습하세요
# (rnn7 함수에 전달된 것과 동일한 형태의 문자열 배열을 만들면 됩니다)


def make_data(long_text, seq_len=20):
    lb = preprocessing.LabelBinarizer()
    lb.fit(list(long_text))

    onehot = lb.transform(list(long_text))

    x = onehot[:-1]
    y = np.argmax(onehot[1:], axis=1)

    rng = [(i, i+seq_len) for i in range(len(long_text) - seq_len)]
    print(len(long_text), rng[-1])      # 171 (150, 170)

    xx = [x[s:e] for s, e in rng]
    yy = [y[s:e] for s, e in rng]
    print(len(xx))                      # 151

    # print(lb.classes_)
    # exit(-1)
    return np.float32(xx), tf.constant(np.int32(yy)), lb.classes_


def rnn_final(long_text, n_iteration=100):
    x, y, vocab = make_data(long_text)

    batch_size, seq_length, n_classes = x.shape     # (151, 20, 26)
    hidden_size = 7
    # cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
    # outputs, _states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

    cells = [tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size) for _ in range(2)]
    multi = tf.nn.rnn_cell.MultiRNNCell(cells)
    outputs, _states = tf.nn.dynamic_rnn(multi, x, dtype=tf.float32)

    z = tf.layers.dense(inputs=outputs, units=n_classes, activation=None)

    w = tf.ones([batch_size, seq_length])
    loss = tf.contrib.seq2seq.sequence_loss(targets=y, logits=z, weights=w)

    # optimizer = tf.train.GradientDescentOptimizer(0.1)
    optimizer = tf.train.AdamOptimizer(0.01)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(n_iteration):
        sess.run(train)
        c = sess.run(loss)
        print(i, c)

    # 문제
    # 예측한 결과를 출력하세요
    # (윗줄에는 정답, 아랫줄에는 예측한 결과)
    preds = sess.run(z)
    preds_arg = np.argmax(preds, axis=2)
    # print(preds_arg.shape)          # (151, 20)
    # print(preds_arg[0])
    print(long_text)
    print('*' + ''.join(vocab[preds_arg[0]]), end='')

    for p in preds_arg[1:]:
        last = p[-1]
        print(vocab[last], end='')

    sess.close()


long_text = (
    "If you want to build a ship, don't drum up people to collect wood"
    " and don't assign them tasks and work, but rather teach them"
    " to long for the endless immensity of the sea."
)

rnn_final(long_text, n_iteration=300)

# "If you want to build a ship, don't drum up people to collect wood"
#  If you wan
#           nt to buil
#  If you wan
#   f you want
#     you want
#     you want t

# rnn7(['If you wan', 'nt to buil', ' to build'])
