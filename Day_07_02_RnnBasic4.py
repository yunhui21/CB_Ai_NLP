# Day_07_02_RnnBasic4.py

import tensorflow as tf
import numpy as np
from sklearn import preprocessing


# 문제 1
# 문자열을 전달 받아서  x, y 데이터를 생성하는 함수를 만드세요.
# preprocessing을 사용하지 않습니다.

# 문제 2
# 정확하게 예했는지 알 수 있게 예측한 문자열ㅇ르 출력학세요.

# 문제 3
# preprocessing의 코드를 사용해서 x, y 데이터를 생성하는 함수를 만드세요.

def make_data_1():
    x = [[0, 0, 0, 0, 0, 1],  # t
         [1, 0, 0, 0, 0, 0],  # e
         [0, 1, 0, 0, 0, 0],  # n
         [0, 0, 0, 0, 1, 0],  # s
         [0, 0, 1, 0, 0, 0]]  # o

    y = [0, 1, 4, 2, 3]  # argmax 결과값

    y = np.int32([y])  # 1차원을 2차원으로
    x = np.float32([x])  # 2차원을 3차원으로 변환
    y = tf.constant(y)

    return x, y

def make_data_2(origin):

    word = sorted(set(origin))
    print(word) # ['e', 'n', 'o', 'r', 's', 't']

    # idx2chr = {i:ch for i, ch in enumerate(word)}
    # print(idx2chr) -> {0: 'e', 1: 'n', 2: 'o', 3: 'r', 4: 's', 5: 't'}

    idx2chr = word      # 타입이 다르다.
    chr2idx = {ch:i for i, ch in enumerate(word)}
    # print(chr2idx) -> {'e': 0, 'n': 1, 'o': 2, 'r': 3, 's': 4, 't': 5}

    word_idx = [chr2idx[k] for k in origin]
    # print(word_idx)  ->  [5, 0, 1, 4, 2, 3]

    x = word_idx[:-1]
    y = word_idx[1:]
    # print(x, y) -> [5, 0, 1, 4, 2] [0, 1, 4, 2, 3] y2차원, x는 원핫벡터 후 3차원

    eye = np.eye(len(word), dtype=np.int32)
    # print(eye)
    # [[1 0 0 0 0 0]
    #  [0 1 0 0 0 0]
    #  [0 0 1 0 0 0]
    #  [0 0 0 1 0 0]
    #  [0 0 0 0 1 0]
    #  [0 0 0 0 0 1]]

    xx = eye[x]
    # print(xx)
    # [[0 0 0 0 0 1]
    #  [1 0 0 0 0 0]
    #  [0 1 0 0 0 0]
    #  [0 0 0 0 1 0]
    #  [0 0 1 0 0 0]]

    return np.float32([xx]), tf.constant(np.int32([y])), np.array(idx2chr)

def make_data_3(origin):
    lb = preprocessing.LabelBinarizer()
    word = lb.fit_transform(list(origin))
    # print(word)

    x = word[:-1]
    y = word[1:]
    y = np.argmax(y, axis= 1)
    # print(y)
    
    return np.float32([x]), tf.constant(np.int32([y])), lb.classes_

def rnn4(word, n_iteration=100 ):
    x, y, vocab = make_data_3(word)

    batch_size, seq_length, n_classes = x.shape

    hidden_size = 7
    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)  #
    outputs, _states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    print(outputs.shape)  # (1, 5, 7)

    # z = tf.layers.dense(inputs=outputs[0], units=6, activation=None)
    # print(z.shape)

    z = tf.layers.dense(inputs=outputs, units=n_classes, activation=None)
    print(z.shape)

    # hx = tf.nn.softmax(z)  # (5,6)
    w =tf.ones([batch_size, seq_length])
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
        print(i, c, preds_arg, ''.join(vocab[preds_arg]))

    sess.close()



def rnn4_3d():

    a = np.array([[0,1,2], [6,4,5]])
    b = np.array([[0, 1], [2, 3], [4, 5]])
    # print(np.dot(a,b))

    aa = np.array([a, a])
    bb = np.array([b, b])
    # print(np.dot(aa, bb))
    # print(np.dot(aa, bb).shape)

    ta = tf.Variable(a)
    tb = tf.Variable(b)

    taa = tf.Variable(aa)
    tbb = tf.Variable(bb)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    hx = sess.run(tf.matmul(ta, tb))
    print(hx)
    print(hx.shape)

    hx = sess.run(tf.matmul(taa, tbb))
    print(hx)
    print(hx.shape)

    sess.close()
rnn4('tensor')
#nn4('preprocessing', n_iteration=300)
# rnn4('hello')


