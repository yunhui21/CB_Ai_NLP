# Day_24_01_Wine.py
import numpy as np
from sklearn import model_selection, preprocessing
import tensorflow as tf


# 문제 1 (소프트맥스 리그레션)
# wine 데이터셋을 x_train, x_test, y_train, y_test로
# 반환하는 함수를 만드세요

# 문제 2
# 학습하고 정확도를 구하세요 (미니배치/멀티레이어 사용 금지)

# 문제 3
# 7개로 구성된 앙상블 모델을 만드세요

def show_accuracy_sparse(preds, labels):
    preds_arg = np.argmax(preds, axis=1)
    print('acc :', np.mean(preds_arg == labels))


def get_data():
    wine = np.loadtxt('data/wine.data', delimiter=',')
    print(wine.shape)           # (178, 14)
    print(wine.dtype)           # float64

    x = wine[:, 1:]
    y = wine[:, 0]

    y -= 1                      # (1, 2, 3) -> (0, 1, 2)
    y = np.int32(y)             # float64 -> int32

    print(x.shape, y.shape)     # (178, 13) (178,)

    return model_selection.train_test_split(x, y, train_size=0.7)


def model_wine(x_train, x_test, y_train, y_test):
    n_features = x_train.shape[1]
    n_classes = 3
    # w = tf.Variable(tf.random_uniform([n_features, n_classes]))
    # b = tf.Variable(tf.random_uniform([n_classes]))

    w = tf.get_variable('w', shape=[n_features, n_classes],
                        initializer=tf.glorot_uniform_initializer)
    b = tf.Variable(tf.zeros([n_classes]))

    ph_x = tf.placeholder(tf.float32)

    z = tf.matmul(ph_x, w) + b
    hx = tf.nn.softmax(z)

    loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_train, logits=z)
    loss = tf.reduce_mean(loss_i)

    # optimizer = tf.train.GradientDescentOptimizer(0.0001)
    optimizer = tf.train.AdamOptimizer(0.01)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train, {ph_x: x_train})
        # print(i, sess.run(loss, {ph_x: x_train}))

    preds_test = sess.run(hx, {ph_x: x_test})
    show_accuracy_sparse(preds_test, y_test)

    sess.close()
    return preds_test


x_train, x_test, y_train, y_test = get_data()

results = np.zeros([len(y_test), 3])
for i in range(7):
    with tf.variable_scope(str(i)):
        preds = model_wine(x_train, x_test, y_train, y_test)
        results += preds

print('-' * 30)
show_accuracy_sparse(results, y_test)
