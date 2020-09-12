# Day_07_05_preprocessing.py
from sklearn import preprocessing, impute
import numpy as np


# 단순 인코딩
def label_encoder():
    # a = [3, 1, 2, 7]
    a = ['brazil', 'cuba', 'bali', 'swiss', 'bali', 'swiss', ]

    enc = preprocessing.LabelEncoder()
    enc.fit(a)
    t = enc.transform(a)
    print(t)
    print(enc.classes_)
    print(enc.classes_[t])
    print(enc.classes_[[1, 2, 0, 3, 0, 3]])

    print(enc.inverse_transform(t))

    # 문제
    # 인코딩 결과를 원핫 벡터로 변환하세요
    # eye = np.eye(4, dtype=np.int32)
    eye = np.eye(len(enc.classes_), dtype=np.int32)
    print(eye)
    print(eye[0], eye[-1])
    print(eye[[0, -1]])
    print(eye[t])


# onehot 인코딩
def label_binarizer():
    # a = [3, 1, 2, 7]
    a = ['brazil', 'cuba', 'bali', 'swiss', 'bali', 'swiss', ]

    enc = preprocessing.LabelBinarizer()
    enc.fit(a)

    print(enc.classes_)

    t = enc.transform(a)
    print(t)
    print(enc.inverse_transform(t))

    # 문제
    # 넘파이를 사용해서 inverse_transform 함수를 만드세요
    idx = np.argmax(t, axis=1)
    print(idx)
    print(enc.classes_[idx])


def add_dummy_feature():
    x = [[0, 1, 2],
         [5, 7, 9]]

    x = preprocessing.add_dummy_feature(x)
    print(x)
    print(type(x))
    print(x.dtype)

    # 문제
    # 2차원 배열의 첫 번째 행에 더미 피처를 추가하세요
    # (np.transpose 함수 사용)
    x = np.transpose(x)
    print(x)

    x = preprocessing.add_dummy_feature(x)
    print(x)

    x = np.transpose(x)
    print(x)


# standarization (표준화)
def standard_scale():
    x = [[-3, 7],
         [2, 3],
         [5, 1]]

    scaler = preprocessing.StandardScaler()
    scaler.fit(x)
    print(scaler.transform(x))
    print(preprocessing.scale(x))


def minmax_scale():
    x = [[-3, 9],
         [1, 5],
         [5, 1]]

    scaler = preprocessing.MinMaxScaler()
    scaler.fit(x)
    print(scaler.transform(x))
    print(preprocessing.minmax_scale(x))


def imputer():
    # 6 = (3 + 9) / 2
    # 4 = (1 + 4 + 7) / 3
    x = [[3, 7],
         [np.nan, 4],
         [9, 1]]

    imp = impute.SimpleImputer()
    imp.fit(x)
    print(imp.transform(x))

    # 문제
    # 각 컬럼마다 결측치가 들어있는 데이터를 만들어서 변환하세요
    # 기존에 만들었던 imp를 사용합니다
    print(imp.transform([[np.nan, np.nan]]))
    print(imp.strategy)
    print(imp.missing_values)
    print(imp.statistics_)


# label_encoder()
# label_binarizer()
# add_dummy_feature()
# standard_scale()
# minmax_scale()
imputer()

# 'bali'   -> 0
# 'brazil' -> 1
# 'cuba'   -> 2
# 'swiss'  -> 3

# 'bali'   -> 0 0
# 'brazil' -> 0 1
# 'cuba'   -> 1 0
# 'swiss'  -> 1 1

# onehot vector
# 'bali'   -> 1 0 0 0
# 'brazil' -> 0 1 0 0
# 'cuba'   -> 0 0 1 0
# 'swiss'  -> 0 0 0 1
