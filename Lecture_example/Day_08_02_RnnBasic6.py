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
        #
        # print(i, c)

        preds = sess.run(z)
        preds_arg = np.argmax(preds, axis=2)

        # 문제
        # 컴프리헨션을 사용해서 출력 결과를 예쁘게 만드세요
        print(i, c, [''.join(vocab[p]) for p in preds_arg])

    sess.close()


rnn6(['tensor', 'coffee', 'clouds'])
#          or     of         ou   => 'o' 다음 글자를 예측하기 어렵다
