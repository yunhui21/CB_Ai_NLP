# Day_14_01_ReviewBreastCancer.py
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import model_selection, preprocessing

# 문제
# 지난 과정에서 수업했던 Breast Cancer 데이터 코드를
# 케라스 버전으로 변환하세요


def get_data():
    #    1. Sample code number            id number
    #    2. Clump Thickness               1 - 10
    #    3. Uniformity of Cell Size       1 - 10
    #    4. Uniformity of Cell Shape      1 - 10
    #    5. Marginal Adhesion             1 - 10
    #    6. Single Epithelial Cell Size   1 - 10
    #    7. Bare Nuclei                   1 - 10
    #    8. Bland Chromatin               1 - 10
    #    9. Normal Nucleoli               1 - 10
    #   10. Mitoses                       1 - 10
    #   11. Class:                        (2 for benign, 4 for malignant)
    names = ['code', 'Clump', 'Size', 'Shape', 'Adhesion',
             'Epithelial', 'Nuclei', 'Chromatin', 'Nucleoli',
             'Mitoses', 'Class']
    bc = pd.read_csv('data/breast-cancer-wisconsin.data',
                     header=None,
                     names=names)
    print(bc)
    bc.info()

    counts = bc.Nuclei.value_counts()
    print(counts)

    most_freq = counts.index[0]

    # print(bc[6])
    # bc.drop([6], axis=1, inplace=True)

    enc = preprocessing.LabelEncoder()
    y = enc.fit_transform(bc['Class'])
    y = y.reshape(-1, 1)
    y = np.float32(y)           # int -> float

    nuclei = bc.Nuclei.values
    print(type(nuclei), nuclei.dtype)
    print(nuclei[:5])
    print(set(nuclei))
    print(np.unique(nuclei))

    # 2번
    # equals = (nuclei == '?')        # [False True True ... False]
    # nuclei[nuclei == '?'] = '0'
    nuclei[nuclei == '?'] = str(most_freq)
    nuclei = np.int64(nuclei)
    print(np.unique(nuclei))

    # 1번
    # temp = [i if i != '?' else '0' for i in nuclei]
    # temp = np.int64(temp)
    # print(np.unique(temp))

    # print(bc.Nuclei)
    bc.drop(['code', 'Nuclei', 'Class'], axis=1, inplace=True)

    # 1번
    # bc['Nuclei'] = temp
    # bc.info()

    x = bc.values

    # 2번
    x = np.hstack([x, nuclei.reshape(-1, 1)])
    print(x.shape, y.shape)     # (699, 8) (699, 1)
    print(x.dtype)              # np.int64

    # 원본 코드에서는 int64로 동작했지만, 케라스에서는 동작 안함
    x = np.float32(x)

    return model_selection.train_test_split(x, y, train_size=0.7)


def show_accuracy(preds, labels):
    preds = preds.reshape(-1)

    bools = np.int32(preds > 0.5)
    y_bools = np.int32(labels.reshape(-1))

    print('acc :', np.mean(bools == y_bools))


def model_breast_cancer_missing(x_train, x_test, y_train, y_test):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)
        # tf.keras.layers.Dense(1, activation='sigmoid')
        # tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.1),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['acc'])
    model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=0)
    # print('acc :', model.evaluate(x_test, y_test, verbose=0))

    preds = model.predict(x_test, verbose=0)
    show_accuracy(preds, y_test)

    return preds


def model_breast_cancer_missing_softmax_sparse(x_train, x_test, y_train, y_test):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(2, activation=tf.keras.activations.softmax)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.1),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])
    model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=0)
    # print('acc :', model.evaluate(x_test, y_test, verbose=0))

    preds = model.predict(x_test, verbose=0)
    preds_arg = np.argmax(preds, axis=1)

    show_accuracy(preds_arg, y_test)
    return preds_arg.reshape(-1, 1)


def model_breast_cancer_missing_softmax_dense(x_train, x_test, y_train, y_test):
    # 원핫 벡터로 변환되지 않음 (2가진인 경우는 단순 인코딩 적용)
    # y_train = preprocessing.LabelBinarizer().fit_transform(y_train)
    # y_test = preprocessing.LabelBinarizer().fit_transform(y_test)

    # np.eye 함수와 동일
    onehot = np.identity(2, dtype=np.float32)
    y_train = onehot[np.int32(y_train.reshape(-1))]
    y_test = onehot[np.int32(y_test.reshape(-1))]
    # print(y_train[:5])

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(2, activation=tf.keras.activations.softmax)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.1),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=['acc'])
    model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=0)
    # print('acc :', model.evaluate(x_test, y_test, verbose=0))

    preds = model.predict(x_test, verbose=0)
    preds_arg = np.argmax(preds, axis=1)
    ytest_arg = np.argmax(y_test, axis=1)

    show_accuracy(preds_arg, ytest_arg)
    return preds_arg.reshape(-1, 1)


x_train, x_test, y_train, y_test = get_data()
print(x_train.dtype, y_train.dtype)

results = np.zeros(y_test.shape)
for i in range(7):
    # preds = model_breast_cancer_missing(x_train, x_test, y_train, y_test)
    # preds = model_breast_cancer_missing_softmax_sparse(x_train, x_test, y_train, y_test)
    preds = model_breast_cancer_missing_softmax_dense(x_train, x_test, y_train, y_test)
    results += preds

print('-' * 30)
show_accuracy(results / 7, y_test)

















