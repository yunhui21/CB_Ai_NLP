# Day_08_01_RnnBasic5.py
import tensorflow as tf
import numpy as np
from sklearn import preprocessing

# 문제 1
# train과 test, 두 개의 데이터셋을 사용하도록 이전 코드를 수정하세요
# (place holder를 사용하세요)

# 문제 2
# 길이가 다른 문자열에 대해 동작하도록 수정하세요
# (원래 길이가 5였다면 3글자 전달)


def make_data(word_train, word_test):
    # 패딩 추가
    # word_test += '  '
    word_test += ' ' * (5 - len(word_test))
    if len(word_test) > 5:
        word_test = word_test[:5]

    lb = preprocessing.LabelBinarizer()
    word = lb.fit_transform(list(word_train))

    x = word[:-1]
    y = word[1:]
    y = np.argmax(y, axis=1)

    # 여기서는 -1번째를 제외할 필요가 없다
    # 'tenso' -> ['t', 'e', 'n', 's', 'o']
    x_test = lb.transform(list(word_test))

    return np.float32([x]), tf.constant(np.int32([y])), lb.classes_, np.float32([x_test])


def rnn5(word_train, word_test, n_iteration=100):
    x, y, vocab, x_test = make_data(word_train, word_test)

    ph_x = tf.placeholder(tf.float32, shape=(None, 5, 6))

    batch_size, seq_length, n_classes = x.shape     # (1, 5, 6)
    hidden_size = 7
    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
    outputs, _states = tf.nn.dynamic_rnn(cell, ph_x, dtype=tf.float32)
    print(outputs.shape)    # (1, 5, 7)

    z = tf.layers.dense(inputs=outputs, units=n_classes, activation=None)
    print(z.shape)          # (1, 5, 6)

    w = tf.ones([batch_size, seq_length])
    loss = tf.contrib.seq2seq.sequence_loss(targets=y, logits=z, weights=w)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(n_iteration):
        sess.run(train, {ph_x: x})
        c = sess.run(loss, {ph_x: x})

        preds = sess.run(z, {ph_x: x_test})
        preds_arg = np.argmax(preds, axis=2)
        preds_arg = preds_arg.reshape(-1)
        # 더 긴 문자열이 들어온다면, 직접 처리해 봅니다
        preds_arg = preds_arg[:len(word_test)]
        print(i, c, preds_arg, ''.join(vocab[preds_arg]))

    sess.close()


# rnn5('tensor', 'tenso')
# rnn5('tensor', 'osnet')
rnn5('tensor', 'neo')

