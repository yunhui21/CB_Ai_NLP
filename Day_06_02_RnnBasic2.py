# Day_06_02_RnnBasic2.py
import tensorflow as tf
import numpy as np

# 문제 1.
# rcc 셀 다음에 dense(matmul) 레이어가 오도록 코드를 수정하세요.
# 이 때, hidden_size의 크기는 2로 설정합니다.

#문제 2
# 원핫 벡터로 된 y를 단순 인코딩 형태로 바꾸어서 동작하도록 수정하세요.
# 몇 개나 맞았는지 알려주세요.

def rnn2_1():
    x = [[0, 0, 0, 0, 0, 1],  # t
         [1, 0, 0, 0, 0, 0],  # e
         [0, 1, 0, 0, 0, 0],  # n
         [0, 0, 0, 0, 1, 0],  # s
         [0, 0, 1, 0, 0, 0]]  # o
    y = [[1, 0, 0, 0, 0, 0],  # e
         [0, 1, 0, 0, 0, 0],  # n
         [0, 0, 0, 0, 1, 0],
         [0, 0, 1, 0, 0, 0],
         [0, 0, 0, 1, 0, 0]]

    x = np.float32([x])     #2차원을 3차원으로 변환

    hidden_size = 7
    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)   #
    outputs, _states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

    #dynamic_rnn이 리턴하는것이 무엇인지 출력
    # outputs :
    # _states : weight를 리턴하기위해서

    print(outputs.shape)       #(1, 5, 2)

    w = tf.Variable(tf.random.uniform([hidden_size,6]))
    b = tf.Variable(tf.random.uniform([6])) # bias

    # (5, 6) = (5, 2) @ (2, 6)
    # 2차원의 확장
    z  = tf.matmul(outputs[0], w) + b
    hx = tf.nn.softmax(z) # (5,6)

    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer =tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train)
        c = sess.run(loss)
        preds = sess.run(z)
        preds_arg = np.argmax(preds, axis = 1)
        print(i, c, preds_arg)

    sess.close()
    # 99 0.7226122 [0 1 4 2 3]
    # http://210.125.150.125/NL/ 자료 확인


def rnn2_2():
    x = [[0, 0, 0, 0, 0, 1],  # t
         [1, 0, 0, 0, 0, 0],  # e
         [0, 1, 0, 0, 0, 0],  # n
         [0, 0, 0, 0, 1, 0],  # s
         [0, 0, 1, 0, 0, 0]]  # o
    # y = [[1, 0, 0, 0, 0, 0],  # e
    #      [0, 1, 0, 0, 0, 0],  # n
    #      [0, 0, 0, 0, 1, 0],  # s
    #      [0, 0, 1, 0, 0, 0],  # o
    #      [0, 0, 0, 1, 0, 0]]  # r 한눈에 들어오기 어렵다.

    y = [0, 1, 4, 2, 3]  # argmax 결과값
    x = np.float32([x])  # 2차원을 3차원으로 변환

    hidden_size = 7
    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)  #
    outputs, _states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

    # dynamic_rnn이 리턴하는것이 무엇인지 출력
    # outputs :
    # _states : weight를 리턴하기위해서

    print(outputs.shape)  # (1, 5, 2)

    w = tf.Variable(tf.random.uniform([hidden_size, 6]))
    b = tf.Variable(tf.random.uniform([6]))  # bias

    # (5, 6) = (5, 2) @ (2, 6)
    # 2차원의 확장
    z = tf.matmul(outputs[0], w) + b
    hx = tf.nn.softmax(z)  # (5,6)

    loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train)
        c = sess.run(loss)
        preds = sess.run(z)
        preds_arg = np.argmax(preds, axis=1)
        print(i, c, preds_arg)
    print(preds_arg == y)
    print('acc:', np.mean(preds_arg == y))
    sess.close()
    # 99 0.7226122 [0 1 4 2 3]
    # http://210.125.150.125/NL/ 자료 확인

# rnn2_1()
rnn2_2()

