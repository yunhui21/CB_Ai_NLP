# Day_13_02_preprocessing.py
from sklearn import preprocessing, impute
import numpy as np


def add_dummy_feature():
    a = [[1, 3], [5, 7]]
    print(a)

    b = preprocessing.add_dummy_feature(a)
    print(b)
    print(type(b))
    print(b.dtype)


def binarizer():
    a = [[-1, 3, -2],
         [5, -7, -4]]

    b = preprocessing.binarize(a)
    print(b)
    print(preprocessing.binarize(a, threshold=-2))

    bin = preprocessing.Binarizer()
    print(bin.transform(a))


def imputer():
    # 4 = (1 + 7) / 2
    # 5 = (2 + 4 + 9) / 3
    x = [[1, 2],
         [np.nan, 4],
         [7, 9]]

    imp = impute.SimpleImputer(strategy='mean')
    imp.fit(x)

    print(imp.transform(x))

    x2 = [[1, np.nan],
          [np.nan, np.nan],
          [np.nan, 9]]
    print(imp.transform(x2))

    print(imp.strategy)


def label_binarizer():
    x = [1, 2, 6, 2, 4]

    lb1 = preprocessing.LabelBinarizer()
    lb1.fit(x)

    print(lb1.transform(x))
    print(lb1.transform([2, 1]))

    print(lb1.classes_)

    # 문제
    # 변환 결과를 원래 상태로 복원하세요
    y = lb1.transform(x)
    print(y)
    print(np.argmax(y, axis=1))
    print(lb1.classes_[np.argmax(y, axis=1)])

    print(lb1.inverse_transform(y))
    print('-' * 30)

    cities = ['bali', 'brazil']
    print(preprocessing.LabelBinarizer().fit_transform(cities))

    cities = ['bali', 'brazil', 'cuba', 'cuba']
    print(preprocessing.LabelBinarizer().fit_transform(cities))


def label_encoder():
    cities = ['bali', 'brazil', 'cuba', 'cuba']

    le = preprocessing.LabelEncoder().fit(cities)
    print(le.transform(cities))

    print(le.classes_)

    # 문제
    # 변환 결과를 원핫 벡터로 바꾸세요
    y = le.transform(cities)

    n_classes = len(le.classes_)
    onehot = np.eye(n_classes, dtype=np.int32)
    print(onehot)
    print(onehot[y])

    print(le.inverse_transform(y))
    print(le.classes_[y])


def minmax_scale():
    x = [[1, -1, 5],
         [2, 0, -4],
         [0, 1, -10]]

    scaler = preprocessing.MinMaxScaler()
    scaler.fit(x)

    print(scaler.transform(x))

    print(scaler.data_max_)
    print(scaler.data_min_)
    print(scaler.data_range_)
    print(scaler.scale_)

    # 문제
    # 아래 수식을 사용해서 원본 데이터를 스케일링하세요
    # 수식
    # (X - X의 최소값) / (X의 최대값 - X의 최소값)
    mx = np.max(x, axis=0)
    mn = np.min(x, axis=0)

    print(mx, mn)
    print((x - mn) / (mx - mn))


def standard_scale():
    x = [[1, -1, 5],
         [2, 0, -4],
         [0, 1, -10]]

    scaler = preprocessing.StandardScaler()
    scaler.fit(x)

    print(scaler.transform(x))


# add_dummy_feature()
# binarizer()
imputer()
# label_binarizer()
# label_encoder()
# minmax_scale()
# standard_scale()
