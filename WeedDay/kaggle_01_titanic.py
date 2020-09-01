# kaggle_01_titanic.py
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import preprocessing

# 문제 1
# 첫 번째 모델을 만들어서 리더보드에 등록하세요 (76.5%)


def get_data_train():
    titanic = pd.read_csv('kaggle/titanic_train.csv')

    titanic.drop(['Sex', 'Age', 'Cabin', 'Embarked',
                  'Name', 'PassengerId', 'Ticket'],
                 axis=1, inplace=True)
    titanic.info()

    x = titanic.values[:, 1:]
    y = titanic.values[:, :1]

    return x, np.float32(y)


def get_data_test():
    titanic = pd.read_csv('kaggle/titanic_test.csv')

    ids = titanic.PassengerId.values

    titanic.drop(['Sex', 'Age', 'Cabin', 'Embarked',
                  'Name', 'PassengerId', 'Ticket'],
                 axis=1, inplace=True)

    x = titanic.values

    return x, ids


def make_submission(file_path, ids, preds):
    f = open(file_path, 'w', encoding='utf-8')

    print('PassengerId,Survived', file=f)
    for i in range(len(ids)):
        result = int(preds[i] > 0.5)
        print('{},{}'.format(ids[i], result), file=f)

    f.close()


def model_titanic(x_train, x_test, y_train):
    name = 'w' + str(np.random.rand())
    w = tf.get_variable(name, shape=[x_train.shape[1], 1],
                        initializer=tf.glorot_uniform_initializer)
    b = tf.Variable(tf.zeros([1]))

    ph_x = tf.placeholder(tf.float32)

    z = tf.matmul(ph_x, w) + b
    hx = tf.sigmoid(z)

    loss_i = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=y_train, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.AdamOptimizer(0.001)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train, {ph_x: x_train})
        print(i, sess.run(loss, {ph_x: x_train}))

    preds = sess.run(hx, {ph_x: x_test})
    sess.close()

    return preds.reshape(-1)


x_train, y_train = get_data_train()
x_test, ids = get_data_test()

print(x_train.shape, x_test.shape)  # (891, 4) (418, 4)
print(y_train.shape)                # (891, 1)

preds = model_titanic(x_train, x_test, y_train)
make_submission('kaggle/titanic_submission.csv', ids, preds)
