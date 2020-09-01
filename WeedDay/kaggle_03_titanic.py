# kaggle_03_titanic.py
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import preprocessing, model_selection

# 문제 6
# Age 컬럼(결측치)을 데이터에 추가하세요

# 문제 7
# Embarked와 Fare 컬럼(결측치)을 데이터에 추가하세요
# (최빈값 사용)

# 문제 8
# 기존의 피처를 사용해서 새로운 피처를 추가하세요

# 문제 9
# 앙상블을 적용하세요


def get_data_train():
    titanic = pd.read_csv('kaggle/titanic_train.csv')

    # print(titanic.Sex.unique())
    # print(titanic.Ticket.unique())
    #
    # print(titanic.Sex.value_counts())
    # print(titanic.Ticket.value_counts())

    # print(titanic['Age'].mean())
    # print(titanic['Age'].mode())
    # print(titanic['Age'].median())
    median = titanic['Age'].median()
    titanic.Age.fillna(median, inplace=True)
    titanic.Embarked.fillna('S', inplace=True)

    titanic.info()
    # print(titanic.Age.describe())

    # print(titanic['Embarked'].mode())
    # print(titanic['Embarked'].value_counts())
    # print(titanic['Fare'].value_counts())
    # print(titanic['Fare'].mean())
    # exit(-1)

    le = preprocessing.LabelEncoder()
    sex = le.fit_transform(titanic.Sex)
    # print(sex.shape)                # (891,)

    sex = np.eye(2, dtype=np.float32)[sex]
    # print(sex.shape)                # (891, 2)
    # print(sex[:3])

    lb = preprocessing.LabelBinarizer()
    embarked = lb.fit_transform(titanic.Embarked)

    titanic.drop(['Sex', 'Cabin', 'Embarked',
                  'Name', 'PassengerId', 'Ticket'],
                 axis=1, inplace=True)
    titanic.info()

    x = np.hstack([titanic.values[:, 1:], sex, embarked])
    y = titanic.values[:, :1]

    return x, np.float32(y), median


def get_data_test(median):
    titanic = pd.read_csv('kaggle/titanic_test.csv')

    titanic.Age.fillna(median, inplace=True)
    # titanic.Embarked.fillna('S', inplace=True)  # mode
    titanic.Fare.fillna(32.20, inplace=True)    # mean

    ids = titanic.PassengerId.values

    le = preprocessing.LabelEncoder()
    sex = le.fit_transform(titanic.Sex)
    sex = np.eye(2, dtype=np.float32)[sex]

    lb = preprocessing.LabelBinarizer()
    embarked = lb.fit_transform(titanic.Embarked)

    titanic.drop(['Sex', 'Cabin', 'Embarked',
                  'Name', 'PassengerId', 'Ticket'],
                 axis=1, inplace=True)

    x = np.hstack([titanic.values, sex, embarked])

    return x, ids


# 새로운 컬럼 추가
def get_data_train_added():
    titanic = pd.read_csv('kaggle/titanic_train.csv')

    # print(titanic.Sex.unique())
    # print(titanic.Ticket.unique())
    #
    # print(titanic.Sex.value_counts())
    # print(titanic.Ticket.value_counts())

    # print(titanic['Age'].mean())
    # print(titanic['Age'].mode())
    # print(titanic['Age'].median())
    median = titanic['Age'].median()
    titanic.Age.fillna(median, inplace=True)
    titanic.Embarked.fillna('S', inplace=True)

    titanic.loc[titanic['Age'] <= 15, 'Age'] = 0
    titanic.loc[(titanic['Age'] > 15) & (titanic['Age'] <= 20), 'Age'] = 1
    titanic.loc[(titanic['Age'] > 20) & (titanic['Age'] <= 26), 'Age'] = 2
    titanic.loc[(titanic['Age'] > 26) & (titanic['Age'] <= 28), 'Age'] = 3
    titanic.loc[(titanic['Age'] > 28) & (titanic['Age'] <= 35), 'Age'] = 4
    titanic.loc[(titanic['Age'] > 35) & (titanic['Age'] <= 45), 'Age'] = 5
    titanic.loc[titanic['Age'] > 45, 'Age'] = 6

    titanic['Family'] = titanic['SibSp'] + titanic['Parch'] + 1
    titanic['IsAlone'] = 1
    titanic['IsAlone'].loc[titanic.Family > 1] = 0

    titanic['FareBin'] = pd.qcut(titanic.Fare, 4, labels=[0, 1, 2, 3])
    # titanic['AgeBin'] = pd.cut(titanic.Age, 5, labels=[0, 1, 2, 3, 4])

    titanic.info()
    # print(titanic.Age.describe())

    # print(titanic['Embarked'].mode())
    # print(titanic['Embarked'].value_counts())
    # print(titanic['Fare'].value_counts())
    # print(titanic['Fare'].mean())

    le = preprocessing.LabelEncoder()
    sex = le.fit_transform(titanic.Sex)
    # print(sex.shape)                # (891,)

    sex = np.eye(2, dtype=np.float32)[sex]
    # print(sex.shape)                # (891, 2)
    # print(sex[:3])

    lb = preprocessing.LabelBinarizer()
    embarked = lb.fit_transform(titanic.Embarked)
    pclass = lb.fit_transform(titanic.Pclass)

    titanic.drop(['Sex', 'Cabin', 'Embarked', 'Pclass',
                  'Name', 'PassengerId', 'Ticket'],
                 axis=1, inplace=True)

    titanic.info()
    print(titanic.columns.values)
    # ['Survived' 'Pclass' 'Age' 'SibSp' 'Parch' 'Fare']
    # print(titanic.Pclass.value_counts())
    # exit(-1)

    x = np.hstack([titanic.values[:, 1:], sex, embarked, pclass])
    y = titanic.values[:, :1]

    return x, np.float32(y), median


def get_data_test_added(median):
    titanic = pd.read_csv('kaggle/titanic_test.csv')

    titanic.Age.fillna(median, inplace=True)
    # titanic.Embarked.fillna('S', inplace=True)  # mode
    titanic.Fare.fillna(32.20, inplace=True)    # mean

    titanic.loc[titanic['Age'] <= 15, 'Age'] = 0
    titanic.loc[(titanic['Age'] > 15) & (titanic['Age'] <= 20), 'Age'] = 1
    titanic.loc[(titanic['Age'] > 20) & (titanic['Age'] <= 26), 'Age'] = 2
    titanic.loc[(titanic['Age'] > 26) & (titanic['Age'] <= 28), 'Age'] = 3
    titanic.loc[(titanic['Age'] > 28) & (titanic['Age'] <= 35), 'Age'] = 4
    titanic.loc[(titanic['Age'] > 35) & (titanic['Age'] <= 45), 'Age'] = 5
    titanic.loc[titanic['Age'] > 45, 'Age'] = 6

    titanic['Family'] = titanic['SibSp'] + titanic['Parch'] + 1
    titanic['IsAlone'] = 1
    titanic['IsAlone'].loc[titanic.Family > 1] = 0

    titanic['FareBin'] = pd.qcut(titanic.Fare, 4, labels=[0, 1, 2, 3])

    ids = titanic.PassengerId.values

    le = preprocessing.LabelEncoder()
    sex = le.fit_transform(titanic.Sex)
    sex = np.eye(2, dtype=np.float32)[sex]

    lb = preprocessing.LabelBinarizer()
    embarked = lb.fit_transform(titanic.Embarked)
    pclass = lb.fit_transform(titanic.Pclass)

    titanic.drop(['Sex', 'Cabin', 'Embarked', 'Pclass',
                  'Name', 'PassengerId', 'Ticket'],
                 axis=1, inplace=True)

    x = np.hstack([titanic.values, sex, embarked, pclass])

    return x, ids


def make_submission(file_path, ids, preds):
    f = open(file_path, 'w', encoding='utf-8')

    print('PassengerId,Survived', file=f)
    for i in range(len(ids)):
        result = int(preds[i] > 0.5)
        print('{},{}'.format(ids[i], result), file=f)

    f.close()


def multi_layers_1(ph_x, ph_d):
    w1 = tf.get_variable(str(np.random.rand()),
                         shape=[x_train.shape[1], 7],
                         initializer=tf.glorot_uniform_initializer)
    b1 = tf.Variable(tf.zeros([7]))

    w2 = tf.get_variable(str(np.random.rand()),
                         shape=[7, 4],
                         initializer=tf.glorot_uniform_initializer)
    b2 = tf.Variable(tf.zeros([4]))

    w3 = tf.get_variable(str(np.random.rand()),
                         shape=[4, 1],
                         initializer=tf.glorot_uniform_initializer)
    b3 = tf.Variable(tf.zeros([1]))

    z1 = tf.matmul(ph_x, w1) + b1
    r1 = tf.nn.relu(z1)
    d1 = tf.nn.dropout(r1, keep_prob=ph_d)

    z2 = tf.matmul(d1, w2) + b2
    r2 = tf.nn.relu(z2)
    d2 = tf.nn.dropout(r2, keep_prob=ph_d)

    z3 = tf.matmul(d2, w3) + b3
    hx = tf.sigmoid(z3)

    return z3, hx


def multi_layers_2(ph_x, ph_d, layers, input_size):
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


def model_titanic(x_train, y_train, x_valid, x_test):
    ph_x = tf.placeholder(tf.float32)
    ph_y = tf.placeholder(tf.float32)
    ph_d = tf.placeholder(tf.float32)   # drop-out

    # z, hx = multi_layers_1(ph_x, ph_d)
    # z, hx = multi_layers_2(ph_x, ph_d,
    #                        layers=[4, 2, 1],
    #                        input_size=x_train.shape[1])
    z, hx = multi_layers_2(ph_x, ph_d,
                           layers=[11, 5, 1],
                           input_size=x_train.shape[1])

    loss_i = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=ph_y, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.AdamOptimizer(0.001)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    epochs = 100
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
        #     print(i, c / n_iteration)

    preds_valid = sess.run(hx, {ph_x: x_valid, ph_d: 1.0})
    preds_test = sess.run(hx, {ph_x: x_test, ph_d: 1.0})
    sess.close()

    return preds_valid.reshape(-1), preds_test.reshape(-1)


# 매개 변수는 모두 1차원
def show_accuracy(preds, labels):
    bools = np.int32(preds > 0.5)
    y_bools = np.int32(labels)

    print('acc :', np.mean(bools == y_bools))


# x, y, median = get_data_train()
# x_test, ids = get_data_test(median)

x, y, median = get_data_train_added()
x_test, ids = get_data_test_added(median)

# scaler = preprocessing.MinMaxScaler()
scaler = preprocessing.StandardScaler()
scaler.fit(x)

x = scaler.transform(x)
x_test = scaler.transform(x_test)

data = model_selection.train_test_split(x, y, train_size=0.7)
x_train, x_valid, y_train, y_valid = data

# print(x_train.shape, x_test.shape)  # (891, 4) (418, 4)
# print(y_train.shape)                # (891, 1)

n_ensembles = 7

results_test_1 = np.zeros([len(x_test)])
for i in range(n_ensembles):
    preds_valid, preds_test = model_titanic(x_train, y_train, x_valid, x_test)
    show_accuracy(preds_valid, y_valid.reshape(-1))

    results_test_1 += preds_test

results_test_2 = np.zeros([len(x_test)])
for i in range(n_ensembles):
    _, preds_test = model_titanic(x, y, x_valid, x_test)
    results_test_2 += preds_test

print('-' * 30)
results_test_1 /= n_ensembles
results_test_2 /= n_ensembles

make_submission('kaggle/titanic_submission_1.csv', ids, results_test_1)
make_submission('kaggle/titanic_submission_2.csv', ids, results_test_2)
