# Day_08_03_RnnBasic6.py

import tensorflow as tf
import numpy as np
from sklearn import preprocessing

# 문제
# 여러 개의 단어(배치 적용)가 입력으로 들올때의 코드로 수정하세요.
# batch_size의 변화를 읽어야 한다.
# placeholder 코드는 삭제해주세요.
# 세번에 걸쳐 세 단어를 훈련해야 한다.
# 하나의 커다란 vocab을 구성한다.


def make_data(word_array):
    # text = word_array[0] + word_array[1] + word_array[2]
    text = ''.join(word_array)

    ib = preprocessing.LabelBinarizer()
    ib.fit(list(text))

    xx, yy = [], []
    for word in word_array:
        origin = ib.transform(list(word))

        x = origin[:-1]
        y = np.argmax(origin[1:], axis=1 )

        xx.append(x)
        yy.append(y)

    return np.float32(xx), tf.constant(np.int32(yy)), ib.classes_

def rnn6(word_array, n_iteration=100):
    x, y, vocab = make_data(word_array)

    batch_size, seq_length, n_classes = x.shape
    hidden_size = 7
    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
    outpus, _states = tf.nn.dynamic_rnn(cell, x, dtype = tf.float32)

    z = tf.layers.dense(inputs=outpus, units=n_classes, activation=None)

    w = tf.ones(batch_size, seq_length)
    loss = tf.contrib.seq2seq.sequence_loss(targets=y, logits=z, weights=w)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(n_iteration):
        sess.run(train)
        c = sess.run(loss)

        print(i, c)

        preds = sess.run(z)
        preds_arg = np.argmax(preds, axis=2)
        print(i, c,[''.join(vocab[p]) for p in preds_arg])
    sess.close()

rnn6(['tensor', 'coffee', 'clouds'])


# def make_data(word_array):
#     # text = word_array[0] + word_array[1] + word_array[2]
#     text = ''.join(word_array)
#
#     lb = preprocessing.LabelBinarizer()
#     # onehot = lb.fit_transform(list(text))     #onehot은 사용하지 않는다.
#     lb.fit(list(text))
#
#     xx, yy = [], []
#     for word in word_array:     #위의 학습결과를 개별 데이터로 반환한다.
#         origin = lb.transform(list(word))
#
#         x = origin[:-1]
#         y = np.argmax(origin[1:], axis=1)
#         # print(x)
#         # print(y)
#
#         xx.append(x)
#         yy.append(y)
#
#     return np.float32(xx), tf.constant(np.int32(yy)), lb.classes_
#
# def rnn6(word_array, n_iteration=100):
#     x, y, vocab = make_data(word_array)
#
#     batch_size, seq_length, n_classes = x.shape     #(3, 5. 11)
#     hidden_size = 7
#     cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)  #
#     outputs, _states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
#
#     z = tf.layers.dense(inputs=outputs, units=n_classes, activation=None)
#
#     w = tf.ones([batch_size, seq_length])
#     loss = tf.contrib.seq2seq.sequence_loss(targets=y, logits=z, weights=w)
#
#     optimizer = tf.train.GradientDescentOptimizer(0.1)
#     train = optimizer.minimize(loss)
#
#     sess = tf.Session()
#     sess.run(tf.global_variables_initializer())
#
#     for i in range(n_iteration):
#         sess.run(train)
#         c = sess.run(loss)
#
#         # print(i, c)
#
#         preds = sess.run(z)
#         preds_arg = np.argmax(preds, axis=2)
#
#
#         # 문제
#         # 컴프리핸션을 사용해서 출력 결과를 예쁘게 만드세요.
#         print(i, c,[''.join(vocab[p]) for p in preds_arg])
#
#         # preds_arg = preds_arg.reshape(-1)
#         # 더 긴 문자열이 들어온다면, 직접 처리해봅니다.
#         # preds_arg = preds_arg[:len(word_test)]
#         # print(i, c, preds_arg, ''.join(vocab[preds_arg]))
#     sess.close()
#
#
# # rnn5('tensor', 'tenso')
# # rnn5('tensor', 'osnet')     #softmax는 다음의 결과를 나타내고, rnn에서는 다른 기억들을 기억해서 추측
# rnn6(['tensor', 'coffee', 'cloud'])




'''
# Day_08_02_RnnBasic6.py
import tensorflow as tf
import numpy as np
from sklearn import preprocessing

# 문제
# 여러 개의 단어(배치 적용)가 입력으로 들어올 때의 코드로 수정하세요
# (앞에서 만들었던 플레이스 홀더 코드는 삭제합니다)


def make_data(word_array):
    # text = word_array[0] + word_array[1] + word_array[2]
    text = ''.join(word_array)

    lb = preprocessing.LabelBinarizer()
    lb.fit(list(text))

    xx, yy = [], []
    for word in word_array:
        origin = lb.transform(list(word))

        x = origin[:-1]
        y = np.argmax(origin[1:], axis=1)
        # print(x)
        # print(y)

        xx.append(x)
        yy.append(y)

    return np.float32(xx), tf.constant(np.int32(yy)), lb.classes_


def rnn6(word_array, n_iteration=100):
    x, y, vocab = make_data(word_array)

    batch_size, seq_length, n_classes = x.shape     # (3, 5, 11)
    hidden_size = 7
    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
    outputs, _states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

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

        print(i, c)

        preds = sess.run(z)
        preds_arg = np.argmax(preds, axis=2)
        print(preds_arg)
        # preds_arg = preds_arg.reshape(-1)
        # # 더 긴 문자열이 들어온다면, 직접 처리해 봅니다
        # preds_arg = preds_arg[:len(word_test)]
        # print(i, c, preds_arg, ''.join(vocab[preds_arg]))

    sess.close()


rnn6(['tensor', 'coffee', 'clouds'])

'''