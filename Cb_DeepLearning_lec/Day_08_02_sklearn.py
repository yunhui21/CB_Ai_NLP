# Day_08_02_sklearn.py
from sklearn import datasets, model_selection, svm, linear_model
import numpy as np

np.set_printoptions(linewidth=1000)


# 문제 1
# load_digits 함수가 반환하는 데이터에 대해 정확도를 계산하세요

# 문제 2
# 마지막 1개를 제외한 데이터로 학습하고, 마지막 1개에 대해 맞았는지 틀렸는지 알려주세요

# 문제 3
# 70%의 데이터로 학습하고 30%의 데이터에 대해 정확도를 예측하세요


def sk_1():
    x, y = datasets.load_iris(return_X_y=True)
    print(x.shape, y.shape)
    print(x[:5])
    print(y[:5])

    clf = svm.SVC()
    clf.fit(x, y)
    print('acc :', clf.score(x, y))

    preds = clf.predict(x)
    print(preds)
    print(y)

    # 문제
    # score 함수가 얘기한 것처럼 정확도를 직접 구하세요
    bools = (preds == y)
    print(bools)
    print('acc :', np.mean(bools))


def sk_2():
    x, y = datasets.load_digits(return_X_y=True)
    print(x.shape, y.shape)     # (1797, 64) (1797,)
    print(x[:5])
    print(y[:5])

    clf = svm.SVC()
    clf.fit(x, y)
    print('acc :', clf.score(x, y))

    preds = clf.predict(x)
    print(preds)
    print(y)

    # 문제
    # score 함수가 얘기한 것처럼 정확도를 직접 구하세요
    bools = (preds == y)
    print(bools)
    print('acc :', np.mean(bools))


def sk_3():
    x, y = datasets.load_digits(return_X_y=True)

    clf = svm.SVC()
    clf.fit(x[:-1], y[:-1])
    # print(clf.score(x[:-1], y[:-1]))

    print(x[-1:].shape)
    print(x[:-1].shape)

    # pred = clf.predict(x[-1])     # error. 차원 다름
    pred = clf.predict(x[-1:])
    print(pred, y[-1])


def sk_4():
    x, y = datasets.load_digits(return_X_y=True)

    n = int(len(x) * 0.7)
    x_train, x_test = x[:n], x[n:]
    y_train, y_test = y[:n], y[n:]

    clf = svm.SVC()
    clf.fit(x_train, y_train)
    print('acc :', clf.score(x_test, y_test))


def sk_5():
    x, y = datasets.load_digits(return_X_y=True)

    n = int(len(x) * 0.7)
    x_train, x_test = x[:n], x[n:]
    y_train, y_test = y[:n], y[n:]

    # clf = svm.SVC()
    clf = linear_model.SGDClassifier()
    clf.fit(x_train, y_train)
    print('acc :', clf.score(x_test, y_test))


# sk_1()
# sk_2()
# sk_3()
# sk_4()
sk_5()
