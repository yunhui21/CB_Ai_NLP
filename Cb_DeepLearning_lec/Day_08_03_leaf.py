# Day_08_03_leaf.py
from sklearn import model_selection, svm, preprocessing
import numpy as np
import pandas as pd

# 문제
# leaf.csv 파일을 읽어서
# 70%의 데이터로 학습하고 30% 데이터로 예측하세요
leaf = pd.read_csv('data/leaf.csv', index_col=0)
print(leaf)

x = leaf.values[:, 1:]
y = preprocessing.LabelEncoder().fit_transform(leaf.species)
# print(y)

x = preprocessing.scale(x)

n = int(len(x) * 0.7)
x_train, x_test = x[:n], x[n:]
y_train, y_test = y[:n], y[n:]

clf = svm.SVC()
clf.fit(x_train, y_train)
print('acc :', clf.score(x_test, y_test))




