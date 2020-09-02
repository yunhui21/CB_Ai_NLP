# Day_08_01_RnnBasic5.py

import tensorflow as tf
import numpy as np
from sklearn import preprocessing


# 문제
# train, test ㄷ 개의 데이터셋을 사용하다록 이전 코들르 수정하세요.
# place holder를 사용하세요.

# 문제
# 길이가 다른 문자열에 대해 동작하도록 수정하세요.
# (원래 길이가 5였다면 3글자 전달)

# ValueError: Cannot feed value of shape (1, 3, 6) for Tensor 'Placeholder:0', which has shape '(?, 5, 6)'
# 들어오는 입력갑의 shape을 수정해주면 될까? ph_x: flaceholder의 값에서 수정.
# 글자수를 무조건 맞춰준다.

def make_data(word_train, word_test):

    word_test += ' '*(5-len(word_test))     #padding 추가 : 학습에 사용하지 않는 임의의 글자를 추가하도록.
    if len(word_test) > 5:
        word_test = word_test[:5]

    lb = preprocessing.LabelBinarizer()
    word = lb.fit_transform(list(word_train))

    x = word[:-1]
    y = word[1:]
    y = np.argmax(y, axis=1)    #word_test를 검증해주어야 한다.

    # 여기서는 -1번째를 베재할 필요가 없다.
    # 'tenso' =>['t','e','n','s','o']
    x_test = lb.transform(list(word_test))  #list

    return np.float32([x]), tf.constant(np.int32([y])), lb.classes_, np.float32([x_test])

def rnn5(word_train, word_test, n_iteration=100):
    x, y, vocab, x_test = make_data(word_train, word_test)

    ph_x = tf.placeholder(tf.float32, shape=(None, 5, 6))

    batch_size, seq_length, n_classes = x.shape
    hidden_size = 7
    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)  #
    outputs, _states = tf.nn.dynamic_rnn(cell, ph_x, dtype=tf.float32)
    print(outputs.shape)  # (1, 5, 7)

    z = tf.layers.dense(inputs=outputs, units=n_classes, activation=None)
    print(z.shape)

    # z = tf.matmul(ph_x, w) + b

    # hx = tf.nn.softmax(z)  # (5,6)
    w = tf.ones([batch_size, seq_length])
    loss = tf.contrib.seq2seq.sequence_loss(targets=y, logits=z, weights=w)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(n_iteration):

        sess.run(train, {ph_x: x})
        c = sess.run(loss, {ph_x:x})

        preds = sess.run(z, {ph_x:x_test})
        preds_arg = np.argmax(preds, axis=2)
        preds_arg = preds_arg.reshape(-1)
        # 더 긴 문자열이 들어온다면, 직접 처리해봅니다.
        preds_arg = preds_arg[:len(word_test)]
        print(i, c, preds_arg, ''.join(vocab[preds_arg]))

    sess.close()


# rnn5('tensor', 'tenso')
# rnn5('tensor', 'osnet')     #softmax는 다음의 결과를 나타내고, rnn에서는 다른 기억들을 기억해서 추측
rnn5('tensor', 'neo')



