# Day_23_01_KerasFunctional.py
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, datasets

# 문제 1
# iris 데이터셋에 대해 7대 3으로 나눠서 학습하세요 (케라스 함수형 적용)

# 문제 2
# 사이킷런에 있는 linnerud 데이터에 대해 함수형으로 예측하세요
# (MAE 측정)


def show_inside(middle_layer):
    weights = middle_layer.weights
    print(type(weights))        # <class 'list'>
    print(len(weights))         # 2

    w, b = weights[0], weights[1]
    print(w.shape, b.shape)     # (4, 7) (7,)
    # print(tf.keras.backend.get_value(w))
    # print(tf.keras.backend.get_value(b))

    print(middle_layer.bias)     # weights[1]과 동일

    # --------------------------------------- #

    # 1.14에서는 동작 안함
    # new_model = tf.keras.Model(model.input, middle_layer.output)
    # new_preds = new_model.predict(x_test)
    # print(new_preds)
    # print(new_preds.shape)


def softmax_iris_1():
    iris = pd.read_csv('data/iris(150).csv', index_col=0)

    y = preprocessing.LabelEncoder().fit_transform(iris.Species)

    iris.drop(['Species'], axis=1, inplace=True)
    x = iris.values

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.7)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(7, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(3, activation=tf.keras.activations.softmax))

    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])

    model.fit(x_train, y_train, epochs=100, verbose=0)
    print('acc :', model.evaluate(x_test, y_test, verbose=0))

    # --------------------------------------- #

    model.summary()

    first_layer = model.get_layer(name='dense')
    # print(type(first_layer))

    show_inside(first_layer)


def softmax_iris_2():
    iris = pd.read_csv('data/iris(150).csv', index_col=0)

    y = preprocessing.LabelEncoder().fit_transform(iris.Species)

    iris.drop(['Species'], axis=1, inplace=True)
    x = iris.values

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.7)

    # 1번
    # input = tf.keras.layers.Input([4])
    # dense1 = tf.keras.layers.Dense(7, activation=tf.keras.activations.relu)(input)
    # dense2 = tf.keras.layers.Dense(3, activation=tf.keras.activations.softmax)(dense1)
    #
    # model = tf.keras.Model(input, dense2)

    # 2번 (사용하지 않는 코드. 중간 레이어 접근 설명)
    input = tf.keras.layers.Input([4])
    dense1 = tf.keras.layers.Dense(7, activation=tf.keras.activations.relu)
    output1 = dense1(input)
    dense2 = tf.keras.layers.Dense(3, activation=tf.keras.activations.softmax)
    output2 = dense2(output1)

    model = tf.keras.Model(input, output2)

    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])

    model.fit(x_train, y_train, epochs=100, verbose=0)
    print('acc :', model.evaluate(x_test, y_test, verbose=0))

    # ------------------------------------- #

    # print(type(dense1))     # <class 'tensorflow.python.keras.layers.core.Dense'>
    show_inside(dense1)


def linnerud_functional():
    def show_difference_all(preds, labels):
        diff = preds - labels
        error = np.mean(np.abs(diff), axis=0)
        print('평균 오차 :', error)

    x, y = datasets.load_linnerud(return_X_y=True)
    # print(x)
    # print(y)

    y0, y1, y2 = y[:, :1], y[:, 1:2], y[:, 2:]

    # 1번
    # input = tf.keras.layers.Input([3])
    #
    # output0 = tf.keras.layers.Dense(1)(input)
    # output1 = tf.keras.layers.Dense(1)(input)
    # output2 = tf.keras.layers.Dense(1)(input)

    # 2번
    input = tf.keras.layers.Input([3])

    output0 = tf.keras.layers.Dense(5)(input)
    output0 = tf.keras.layers.Dense(1)(output0)

    output1 = tf.keras.layers.Dense(5)(input)
    output1 = tf.keras.layers.Dense(1)(output1)

    output2 = tf.keras.layers.Dense(5)(input)
    output2 = tf.keras.layers.Dense(3)(output2)
    output2 = tf.keras.layers.Dense(1)(output2)

    model = tf.keras.Model(input, [output0, output1, output2])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                  loss=tf.keras.losses.mse,
                  metrics=['mae'])

    history = model.fit(x, [y0, y1, y2], epochs=100, verbose=0)
    print('acc :', model.evaluate(x, [y0, y1, y2], verbose=0))
    print(history.history.keys())
    # acc : [15408.501953125,    7399.468, 5178.9263, 2830.108,    73.43773, 70.77467, 40.724648]
    # dict_keys(['loss',
    # 'dense_loss', 'dense_1_loss', 'dense_2_loss',
    # 'dense_mean_absolute_error', 'dense_1_mean_absolute_error', 'dense_2_mean_absolute_error'])

    # 문제
    # show_difference_all 함수를 사용해서 mae를 구하세요
    preds = model.predict(x, verbose=0)
    # print(preds)
    # print(type(preds), len(preds))      # <class 'list'> 3

    show_difference_all(np.hstack(preds), y)

    print(np.array(preds).shape)    # (3, 20, 1)
    print(np.hstack(preds).shape)   # (20, 3)

    show_difference_all(preds[0], y0)
    show_difference_all(preds[1], y1)
    show_difference_all(preds[2], y2)


# softmax_iris_1()
# softmax_iris_2()

linnerud_functional()

# 1번
# 평균 오차 : [101.74553   43.071682  25.855396]

# 2번
# 평균 오차 : [70.74712772]
# 평균 오차 : [13.08983927]
# 평균 오차 : [17.71992168]
