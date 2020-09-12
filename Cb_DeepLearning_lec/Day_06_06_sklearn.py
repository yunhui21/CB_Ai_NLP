# Day_06_06_sklearn.py
from sklearn import datasets


iris = datasets.load_iris()
# print(iris)
print(iris.keys())
# dict_keys(['data', 'target', 'frame',
# 'target_names', 'DESCR', 'feature_names', 'filename'])

print(iris['feature_names'])    # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
print(iris['target_names'])     # ['setosa' 'versicolor' 'virginica']

print(iris['data'])
print(iris['data'][:3])
print(iris['target'])

print(iris['data'].shape)       # (150, 4)
print(iris['target'].shape)     # (150,)

print(iris['data'].dtype)       # float64
print(iris['target'].dtype)     # int32

print(iris['DESCR'])
print('-' * 30)

x, y = datasets.load_iris(return_X_y=True)
print(x.shape, y.shape)     # (150, 4) (150,)


