# Day_14_02_sklearn.py
from sklearn import datasets, svm, model_selection
import numpy as np


def predict_1():
    iris = datasets.load_iris()

    clf = svm.SVC()
    clf.fit(iris['data'], iris['target'])

    y_hats = clf.predict(iris['data'])
    equals = (y_hats == iris['target'])
    print('acc :', np.mean(equals))


# 문제
# digits 데이터셋에서 마지막 데이터를 제외한 데이터로 학습하고
# 마지막 데이터에 대해 결과를 예측하세요
def predict_2():
    digits = datasets.load_digits()

    clf = svm.SVC()
    clf.fit(digits['data'][:-1], digits['target'][:-1])

    print(digits['data'].shape)
    print(digits['data'][-1].shape)
    print(digits['data'][-1:].shape)

    y_hats = clf.predict(digits['data'][-1:])
    print(digits['target'][-1:], y_hats)


# 문제
# 70% 데이터로 학습하고 30% 데이터에 대해 정학도를 예측하세요
def predict_3():
    digits = datasets.load_digits()

    x = digits['data']
    y = digits['target']

    # train_size = int(len(x) * 0.7)
    # x_train, x_test = x[:train_size], x[train_size:]
    # y_train, y_test = y[:train_size], y[train_size:]

    # 75 : 25
    # data = model_selection.train_test_split(x, y)

    # data = model_selection.train_test_split(x, y, train_size=0.7)

    data = model_selection.train_test_split(x, y, shuffle=False)
    x_train, x_test, y_train, y_test = data

    clf = svm.SVC()
    clf.fit(x_train, y_train)

    y_hats = clf.predict(x_test)
    equals = (y_hats == y_test)
    print('acc :', np.mean(equals))

    # ----------------------- #

    clf = svm.SVC(gamma=0.001, C=100)
    clf.fit(x_train, y_train)

    y_hats = clf.predict(x_test)
    equals = (y_hats == y_test)
    print('acc :', np.mean(equals))


# predict_1()
# predict_2()
predict_3()







