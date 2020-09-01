# Day_26_02_Abalone.py
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import model_selection, preprocessing

np.set_printoptions(linewidth=1000)

# 문제 1
# abalone.data 파일을
# x_train, x_test, y_train, y_test로 반환하는 함수를 만들고
# 싱글 모델로 정확도를 구하세요

# 문제 2
# 레이블의 갯수를 3개로 줄여서 정확도를 구하세요

def get_data():
    names = ['Sex', 'Length', 'Diamete', 'Height',
             'Whole', 'Shucked', 'Viscera', 'Shell', 'Rings']

    abalone = pd.read_csv('data/abalone.data',
                       header=None,
                       names=names)
    # print(abalone)
    # print(abalone.Rings.unique())
    # print(sorted(abalone.Rings.unique()))
    # print(abalone.Rings.max())

    enc = preprocessing.LabelBinarizer()
    gender = enc.fit_transform(abalone.Sex)

    # enc = preprocessing.LabelEncoder()
    # y = enc.fit_transform(abalone.Rings)
    y = abalone.Rings.values // 10
    # print(sorted(np.unique(y)))

    abalone.drop(['Sex', 'Rings'], axis=1, inplace=True)
    x = np.hstack([gender, abalone.values])

    print(x.shape, y.shape)     # (4177, 10) (4177,)

    return model_selection.train_test_split(x, y, train_size=0.7)


def show_accuracy_sparse(preds, labels):
    preds_arg = np.argmax(preds, axis=1)
    # y_arg = np.argmax(labels, axis=1)

    equals = (preds_arg == labels)
    print('acc :', np.mean(equals))


def model_abalone_sparse(x_train, x_test, y_train, y_test):
    n_features = x_train.shape[1]
    n_classes = np.max(y_train) + 1
    # print(n_classes)
    # print(sorted(np.unique(y_train)))

    w = tf.Variable(tf.random_uniform([n_features, n_classes]))
    b = tf.Variable(tf.random_uniform([n_classes]))

    ph_x = tf.placeholder(tf.float32)

    z = tf.matmul(ph_x, w) + b
    hx = tf.nn.softmax(z)

    loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_train, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train, {ph_x: x_train})
        # print(i, sess.run(loss, {ph_x: x_train}))

    # ---------------------------- #

    preds_test = sess.run(hx, {ph_x: x_test})
    show_accuracy_sparse(preds_test, y_test)

    sess.close()
    return preds_test


x_train, x_test, y_train, y_test = get_data()

results = np.zeros([len(y_test), np.max(y_train) + 1])
preds_list = []
for i in range(7):
    with tf.variable_scope(str(i)):
        preds = model_abalone_sparse(x_train, x_test, y_train, y_test)
        results += preds
        preds_list.append(np.argmax(preds, axis=1))

print('-' * 30)
show_accuracy_sparse(results, y_test)

# ---------------------------------- #

for preds_arg in preds_list:
    print(preds_arg[:30])

print('-' * 30)
print(y_test[:30])
