# Day_11_03_sklearn.py
from sklearn import datasets


def basci_1():
    iris = datasets.load_iris()
    print(type(iris))
    print(iris.keys())
    # dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])

    print(iris['feature_names'])
    # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

    print(iris['target_names'])
    # ['setosa' 'versicolor' 'virginica']

    print(iris['data'][:5])
    # [[5.1 3.5 1.4 0.2]
    #  [4.9 3.  1.4 0.2]
    #  [4.7 3.2 1.3 0.2]
    #  [4.6 3.1 1.5 0.2]
    #  [5.  3.6 1.4 0.2]]

    print(iris['target'])
    # [0 0 0 0 0 ... 2 2]
    print(iris['target'].shape)
    # (150,)

    print(iris['frame'])
    # None

    print(iris['DESCR'])


# 문제
# digits 데이터셋에 대해 알아보고
# x 데이터의 첫 번째 데이터를 정확하게 출력하세요
def basic_2():
    digits = datasets.load_digits()

    print(digits.keys())
    # dict_keys(['data', 'target', 'frame',
    # 'feature_names', 'target_names', 'images', 'DESCR'])

    print(digits['data'].shape)     # (1797, 64)
    print(digits['images'].shape)   # (1797, 8, 8)

    print(digits['data'][0])
    print(digits['images'][0].reshape(-1))

    print(digits['images'][0])
    print(digits['target'][0])


# basic_1()
basic_2()

