# Day_08_03_RnnBasic7.py
import tensorflow as tf
import numpy as np
from sklearn import preprocessing

# 문제
# 길이가 다른 여러 개의 단어(배치 적용)에 대해 처리하세요


def make_data(word_array):
    max_len = max([len(s) for s in word_array])

    # text = word_array[0] + word_array[1] + word_array[2]
    text = ''.join(word_array)

    lb = preprocessing.LabelBinarizer()
    lb.fit(list(text))

    xx, yy = [], []
    for word in word_array:
        if len(word) < max_len:
            word += ' ' * (max_len - len(word))

        origin = lb.transform(list(word))

        x = origin[:-1]
        y = np.argmax(origin[1:], axis=1)
        # print(x)
        # print(y)

        xx.append(x)
        yy.append(y)

    return np.float32(xx), tf.constant(np.int32(yy)), lb.classes_


def rnn7(word_array, n_iteration=100):
    len_array = [len(s) for s in word_array]
    x, y, vocab = make_data(word_array)

    batch_size, seq_length, n_classes = x.shape     # (3, 5, 11)
    hidden_size = 7
    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
    outputs, _states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32, sequence_length=len_array)

    z = tf.layers.dense(inputs=outputs, units=n_classes, activation=None)

    w = tf.ones([batch_size, seq_length])
    loss = tf.contrib.seq2seq.sequence_loss(targets=y, logits=z, weights=w)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(n_iteration):
        sess.run(train)
        c = sess.run(loss)

        preds = sess.run(z)
        preds_arg = np.argmax(preds, axis=2)

        print(i, c, [''.join(vocab[p[:n-1]]) for p, n in zip(preds_arg, len_array)])

    sess.close()


rnn7(['rainbow', 'sky', 'wave'], n_iteration=300)

