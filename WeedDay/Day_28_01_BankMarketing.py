# Day_28_01_BankMarketing.py
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import model_selection, preprocessing


# 문제 1
# get_data 함수를 만드세요
# 파일이 2개로 구성되어 있으니
# train과 test 파일로 각각 사용하세요

# 문제 2
# 은행 영업 데이터에 대해 정확도를 계산하세요 (85% 이상)

# 문제 3
# 스케일링을 적용하세요

# 문제 4
# Label Encoder를 LabelBinarizer로 바꾸세요

# 문제 5
# 멀티레이어로 수정하세요

def get_data_sparse(file_path):
    bank = pd.read_csv(file_path, delimiter=';')
    # print(bank, end='\n\n')
    # print(bank.describe(), end='\n\n')
    #
    # bank.info()
    # print()

    age = bank.age.values
    balance = bank.balance.values
    day = bank.day.values
    duration = bank.duration.values
    campaign = bank.campaign.values
    pdays = bank.pdays.values
    previous = bank.previous.values

    enc = preprocessing.LabelEncoder()

    job = enc.fit_transform(bank.job)
    marital = enc.fit_transform(bank.marital)
    education = enc.fit_transform(bank.education)
    default = enc.fit_transform(bank.default)
    housing = enc.fit_transform(bank.housing)
    loan = enc.fit_transform(bank.loan)
    contact = enc.fit_transform(bank.contact)
    month = enc.fit_transform(bank.month)
    poutcome = enc.fit_transform(bank.poutcome)
    y = enc.fit_transform(bank.y)

    x = np.transpose([
        age, job, marital, education, default,
        balance, housing, loan, contact, day,
        month, duration, campaign, pdays, previous,
        poutcome])

    # 틀렸음
    # x = preprocessing.minmax_scale(x)
    # x = preprocessing.scale(x)

    y = y.reshape(-1, 1)
    y = np.float32(y)

    print(x.shape, y.shape)     # (4521, 16) (4521, 1)
    return x, y


def get_data_dense(file_path):
    bank = pd.read_csv(file_path, delimiter=';')

    age = bank.age.values
    balance = bank.balance.values
    day = bank.day.values
    duration = bank.duration.values
    campaign = bank.campaign.values
    pdays = bank.pdays.values
    previous = bank.previous.values

    enc = preprocessing.LabelBinarizer()

    job = enc.fit_transform(bank.job)
    marital = enc.fit_transform(bank.marital)
    education = enc.fit_transform(bank.education)
    default = enc.fit_transform(bank.default)
    housing = enc.fit_transform(bank.housing)
    loan = enc.fit_transform(bank.loan)
    contact = enc.fit_transform(bank.contact)
    month = enc.fit_transform(bank.month)
    poutcome = enc.fit_transform(bank.poutcome)

    items = np.transpose([age, balance, day, duration,
                          campaign, pdays, previous])
    x = np.hstack([items, job, marital, education, default,
                   housing, loan, contact, month, poutcome])

    y = preprocessing.LabelEncoder().fit_transform(bank.y)
    y = y.reshape(-1, 1)
    y = np.float32(y)

    print(x.shape, y.shape)     # (4521, 48) (4521, 1)
    return x, y


def show_accuracy(preds, labels):
    preds = preds.reshape(-1)

    bools = np.int32(preds > 0.5)
    y_bools = np.int32(labels.reshape(-1))

    print('acc :', np.mean(bools == y_bools))


def model_bank_marketing(x_train, x_test, y_train, y_test):
    # w = tf.Variable(tf.random_uniform([x_train.shape[1], 1]))

    name = str(np.random.rand())
    w1 = tf.get_variable(name, shape=[x_train.shape[1], 41],
                         initializer=tf.glorot_uniform_initializer)
    b1 = tf.Variable(tf.zeros([41]))

    name = str(np.random.rand())
    w2 = tf.get_variable(name, shape=[41, 31],
                         initializer=tf.glorot_uniform_initializer)
    b2 = tf.Variable(tf.zeros([31]))

    name = str(np.random.rand())
    w3 = tf.get_variable(name, shape=[31, 21],
                         initializer=tf.glorot_uniform_initializer)
    b3 = tf.Variable(tf.zeros([21]))

    name = str(np.random.rand())
    w4 = tf.get_variable(name, shape=[21, 11],
                         initializer=tf.glorot_uniform_initializer)
    b4 = tf.Variable(tf.zeros([11]))

    name = str(np.random.rand())
    w5 = tf.get_variable(name, shape=[11, 1],
                         initializer=tf.glorot_uniform_initializer)
    b5 = tf.Variable(tf.zeros([1]))

    ph_x = tf.placeholder(tf.float32)
    ph_y = tf.placeholder(tf.float32)

    z1 = tf.matmul(ph_x, w1) + b1
    r1 = tf.nn.relu(z1)
    z2 = tf.matmul(r1, w2) + b2
    r2 = tf.nn.relu(z2)
    z3 = tf.matmul(r2, w3) + b3
    r3 = tf.nn.relu(z3)
    z4 = tf.matmul(r3, w4) + b4
    r4 = tf.nn.relu(z4)
    z5 = tf.matmul(r4, w5) + b5

    hx = tf.sigmoid(z5)

    loss_i = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=ph_y, logits=z5)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.AdamOptimizer(0.001)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    epochs = 30
    batch_size = 32
    n_iteration = len(x_train) // batch_size

    indices = np.arange(len(x_train))

    for i in range(epochs):
        c = 0
        for j in range(n_iteration):
            n1 = j * batch_size
            n2 = n1 + batch_size

            part = indices[n1:n2]

            xx = x_train[part]
            yy = y_train[part]

            sess.run(train, {ph_x: xx, ph_y: yy})
            c += sess.run(loss, {ph_x: xx, ph_y: yy})

        print(i, c / n_iteration)
        np.random.shuffle(indices)

        preds = sess.run(hx, {ph_x: x_test})
        show_accuracy(preds, y_test)

    sess.close()

    return preds


x_train, y_train = get_data_sparse('data/bank-full.csv')   # (45211, 16) (45211, 1)
x_test, y_test = get_data_sparse('data/bank.csv')          # (4521, 16) (4521, 1)

# x_train, y_train = get_data_dense('data/bank-full.csv')   # (45211, 48) (45211, 1)
# x_test, y_test = get_data_dense('data/bank.csv')          # (4521, 48) (4521, 1)

scaler = preprocessing.StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

model_bank_marketing(x_train, x_test, y_train, y_test)

