# Day_07_03_sklearn.py
from sklearn import datasets, svm, linear_model
import numpy as np
import csv


def show_ml_1():
    x, y = datasets.load_iris(return_X_y=True)
    print(x.shape, y.shape)  # (150, 4) (150,)

    clf = svm.SVC()
    clf.fit(x, y)
    print('acc :', clf.score(x, y))

    clf = linear_model.SGDClassifier()
    clf.fit(x, y)
    print('acc :', clf.score(x, y))

    preds = clf.predict(x)
    # print(preds)
    # print(y)

    equals = (preds == y)
    # print(equals)
    print('acc :', np.mean(equals))


# 문제
# iris.txt 파일을 읽어서 svm과 linear_model 알고리즘에 적용하세요
def get_data_1():
    f = open('data/iris.csv', 'r', encoding='utf-8')
    f.readline()

    x, y = [], []
    for line in f:
        items = line.strip().split(',')
        items =[float(i) for i in items]
        # print(items)

        x.append(items[:-1])
        y.append(items[-1])

    f.close()
    return np.float32(x), np.int32(y)


def get_data_2():
    f = open('data/iris.csv', 'r', encoding='utf-8')
    f.readline()

    x, y = [], []
    for items in csv.reader(f):
        items =[float(i) for i in items]

        x.append(items[:-1])
        y.append(items[-1])

    f.close()
    return np.float32(x), np.int32(y)


def get_data_3():
    f = open('data/iris.csv', 'r', encoding='utf-8')
    f.readline()

    items = np.array(list(csv.reader(f)))
    # print(items)

    f.close()
    return np.float32(items[:, :-1]), np.int32(items[:, -1])


def show_ml_2():
    # x, y = get_data_1()
    # x, y = get_data_2()
    x, y = get_data_3()
    print(x.shape, y.shape)  # (150, 4) (150,)

    clf = svm.SVC()
    clf.fit(x, y)
    print('acc :', clf.score(x, y))

    clf = linear_model.SGDClassifier()
    clf.fit(x, y)
    print('acc :', clf.score(x, y))


# show_ml_1()
show_ml_2()
