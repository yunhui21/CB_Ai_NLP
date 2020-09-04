# Day_07_03_WordRnn.py
import tensorflow as tf
import numpy as np

# 문제
# 문자열 리스트에 대해 예측하세요


def make_data(origin):
    word = sorted(set(origin))
    # print(word)         # ['five', 'four', 'one', 'seven', 'six', 'three', 'two']

    idx2chr = word
    chr2idx = {ch: i for i, ch in enumerate(word)}
    # print(chr2idx)
    # {'five': 0, 'four': 1, 'one': 2, 'seven': 3, 'six': 4, 'three': 5, 'two': 6}

    word_idx = [chr2idx[t] for t in origin]
    # print(word_idx)       # [2, 6, 5, 1, 0, 4, 3]

    x = word_idx[:-1]
    y = word_idx[1:]
    # print(x, y)           # [2, 6, 5, 1, 0, 4] [6, 5, 1, 0, 4, 3]

    eye = np.eye(len(word), dtype=np.int32)

    xx = eye[x]

    return np.float32([xx]), tf.constant(np.int32([y])), np.array(idx2chr)


def rnn_word(word, n_iteration=100):
    x, y, vocab = make_data(word)

    batch_size, seq_length, n_classes = x.shape     # (1, 5, 6)
    hidden_size = 7
    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
    outputs, _states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    print(outputs.shape)    # (1, 5, 7)

    # z = tf.layers.dense(inputs=outputs[0], units=6, activation=None)
    # print(z.shape)          # (5, 6)

    z = tf.layers.dense(inputs=outputs, units=n_classes, activation=None)
    # z = tf.layers.dense(inputs=outputs, units=6, activation='softmax')
    print(z.shape)          # (1, 5, 6)

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
        preds_arg = preds_arg.reshape(-1)
        # print(i, c, preds_arg, *vocab[preds_arg])
        print(i, c, preds_arg, ' '.join(vocab[preds_arg]))

    sess.close()


# numbers = ['this', 'is', 'a', 'pen', 'he', 'gave', 'me', ]
numbers = ['one', 'two', 'three', 'four', 'five', 'six', 'seven']
rnn_word(numbers, n_iteration=300)

