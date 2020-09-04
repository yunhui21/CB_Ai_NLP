# Day_09_01_RnnBasic8.py

import tensorflow as tf
import numpy as np
from sklearn import preprocessing

# 문제
# 굉장히 긴 문자열을 학습하세요.


def make_data(long_text, seq_len=20):
    lb = preprocessing.LabelBinarizer()
    lb.fit(list(long_text))

    onehot = lb.transform(list(long_text))

    x = onehot[:-1]
    y = np.argmax(onehot[1:], axis = 1)

    rng = [(i, i+seq_len) for i in range(len(long_text)-seq_len)]
    print(len(long_text), rng[-1])

    xx = [x[s:e] for s, e in rng]
    yy = [y[s:e] for s, e in rng]
    print(len(xx))

    # print(lb.classes_)
    # exit(-1)
    return np.float32(xx), tf.constant(np.int32(yy)), lb.classes_


def rnn_final(long_text, n_iteration=100):
    x, y, vocab = make_data(long_text)

    batch_size, seq_length, n_classes = x.shape     # (3, 5, 11)
    # print(x.shape)
    hidden_size = 7
    # cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
    # outputs, _states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

    cell = [tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size) for _in range(2)]
    outputs, _states = tf.nn.dynamic_rnn(multi, x, dtype=tf.float32)

    z = tf.layers.dense(inputs=outputs, units=n_classes, activation=None)

    w = tf.ones([batch_size, seq_length])
    loss = tf.contrib.seq2seq.sequence_loss(targets=y, logits=z, weights=w)

#     optimizer = tf.train.GradientDescentOptimizer(0.1)
    optimizer = tf.train.AdamOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(n_iteration):
        sess.run(train)
        c = sess.run(loss)

        print(i, c)
#
#         preds = sess.run(z)
#         preds_arg = np.argmax(preds, axis=2)
#         print(i, c,[''.join(vocab[p[:n-1]]) for p, n in zip(preds_arg, len_array)])
#         # preds_arg = preds_arg.reshape(-1)
#         # # 더 긴 문자열이 들어온다면, 직접 처리해 봅니다
#         # preds_arg = preds_arg[:len(word_test)]
#         # print(i, c, preds_arg, ''.join(vocab[preds_arg]))
#
#     sess.close()
#
#
long_text = (
    "If you want to build a ship, don't drum up people to collect wood"
    " and don't assign them tasks and work, but rather teach them"
    " to long for the endless immensity of the sea."
    )
#
#
# rnn_final(long_text, n_iteration=300)

#  "If you want to build a ship, don't drum up people to collect wood"
#   If you wan
#            Nt to buil

#   If you wan
#    f you want
#       you want t
#       you want to

# rnn7(['If you wan', 'Nt to buil', 'to build']