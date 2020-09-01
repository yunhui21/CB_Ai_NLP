# kaggle_06_CatInTheDat.py
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import preprocessing, model_selection

# 문제
# cat in the dat 캐글 문제를 풀어보세요
# kaggle_03_titanic.py


def get_data_train():
    cat = pd.read_csv('kaggle/cat_train.csv',
                      index_col=0)
    # cat.info()

    # total = 0
    # for col in cat.columns:
    #     v = cat[col].unique()
    #     total += len(v)
    #     print('{} ({}) : {}'.format(col, len(v), v[:10]))
    # print('total :', total)     # total : 16463

    enc = preprocessing.LabelEncoder()
    y = enc.fit_transform(cat.target)

    cat.drop(['target'], axis=1, inplace=True)

    items = []
    for col in cat.columns:
        items.append(enc.fit_transform(cat[col]))

    x = np.transpose(items)

    # print(x.shape, y.shape)     # (300000, 23) (300000,)
    return x, y.reshape(-1, 1)


def get_data_test():
    cat = pd.read_csv('kaggle/cat_test.csv', index_col=0)

    enc = preprocessing.LabelEncoder()

    items = []
    for col in cat.columns:
        items.append(enc.fit_transform(cat[col]))

    x = np.transpose(items)

    return x, cat.index.values


def make_submission(file_path, ids, preds):
    f = open(file_path, 'w', encoding='utf-8')

    print('id,target', file=f)
    for i, p in zip(ids, preds):
        print('{},{}'.format(i, p), file=f)

    f.close()


def multi_layers(ph_x, ph_d, layers, input_size):
    d = ph_x
    n_features = input_size
    for n_classes in layers:
        w = tf.get_variable(str(np.random.rand()),
                            shape=[n_features, n_classes],
                            initializer=tf.glorot_uniform_initializer)
        b = tf.Variable(tf.zeros([n_classes]))

        z = tf.matmul(d, w) + b

        if n_classes == layers[-1]:     # 1
            break

        r = tf.nn.relu(z)
        d = tf.nn.dropout(r, keep_prob=ph_d)

        n_features = n_classes

    return z, tf.sigmoid(z)


def model_cat_in_the_dat(x_train, y_train, x_valid, x_test):
    ph_x = tf.placeholder(tf.float32)
    ph_y = tf.placeholder(tf.float32)
    ph_d = tf.placeholder(tf.float32)   # drop-out

    z, hx = multi_layers(ph_x, ph_d,
                         layers=[1],
                         input_size=x_train.shape[1])

    loss_i = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=ph_y, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.AdamOptimizer(0.001)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    epochs = 10
    batch_size = 32
    n_iteration = len(x_train) // batch_size

    indices = np.arange(len(x_train))

    for i in range(epochs):
        np.random.shuffle(indices)

        c = 0
        for j in range(n_iteration):
            n1 = j * batch_size
            n2 = n1 + batch_size

            part = indices[n1:n2]

            xx = x_train[part]
            yy = y_train[part]

            sess.run(train, {ph_x: xx, ph_y: yy, ph_d: 0.7})
            c += sess.run(loss, {ph_x: xx, ph_y: yy, ph_d: 0.7})

        # if i % 10 == 0:
        print(i, c / n_iteration)

    preds_valid = sess.run(hx, {ph_x: x_valid, ph_d: 1.0})
    preds_test = sess.run(hx, {ph_x: x_test, ph_d: 1.0})
    sess.close()

    return preds_valid.reshape(-1), preds_test.reshape(-1)


# 매개 변수는 모두 1차원
def show_accuracy(preds, labels):
    bools = np.int32(preds > 0.5)
    y_bools = np.int32(labels)

    print('acc :', np.mean(bools == y_bools))


x, y = get_data_train()
x_test, ids = get_data_test()

data = model_selection.train_test_split(x, y, train_size=0.7)
x_train, x_valid, y_train, y_valid = data

# print(x_train.shape, x_test.shape)  #
# print(y_train.shape)                #

preds_valid, preds_test = model_cat_in_the_dat(x, y, x_valid, x_test)
show_accuracy(preds_valid, y_valid.reshape(-1))

make_submission('kaggle/cat_submission.csv', ids, preds_test)






