# Day_23_01_KerasFunctional.py
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, datasets
# tf.compat.v1.enable_eager_execution()


# 문제
# iris 데이터셋에 대해 케라스 함수형 사용해서 예측해보세요.

# 문제
# linnerud데이터에 대해 함수형으로 예측하세요.
def show_inside(middle_layer):
    weights = middle_layer.weights
    print(type(weights))
    print(len(weights))  # 2

    a, b = weights[0], weights[1]
    print(a.shape, b.shape)  # (4, 7) (7,)
    # print(tf.keras.backend.get_value(a))
    # print(tf.keras.backend.get_value(b))

    print(middle_layer.bias)

    # 1.14에서는 동작안함.
    # new_model = tf.keras.Model(model.input, first_layers.output)
    # new_preds = new_model.predict(x_test)
    # print(new_preds.shape)

def softmax_iris_1():
    iris = pd.read_csv('data/iris(150).csv', index_col=0)
    # print(iris) #[150 rows x 5 columns]

    y = preprocessing.LabelEncoder().fit_transform(iris.Species)

    iris.drop(['Species'], axis=1, inplace=True)

    x = iris.values

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.7)

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(7, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(3, activation = tf.keras.activations.softmax))
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])  #

    model.fit(x_train, y_train, epochs=10, verbose=2)
    print('acc:', model.evaluate(x_test, y_test))

    model.summary()

    first_layers = model.get_layer(name='dense')
    print(type(first_layers))

    return show_inside()

def softmax_iris_2():
    iris = pd.read_csv('data/iris(150).csv', index_col=0)
    # print(iris) #[150 rows x 5 columns]

    y = preprocessing.LabelEncoder().fit_transform(iris.Species)

    iris.drop(['Species'], axis=1, inplace=True)

    x = iris.values

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.7)

    # input = tf.keras.layers.Input([4])
    # dense1 =tf.keras.layers.Dense(7, activation=tf.keras.activations.relu)(input)
    # dense2 =tf.keras.layers.Dense(3, activation= tf.keras.activations.softmax)(dense1)
    #
    # model = tf.keras.Model(input.dase2)
    # 중간레이어 사용을 위해 접금하는거.

    input = tf.keras.layers.Input([4])
    dense1 =tf.keras.layers.Dense(7, activation=tf.keras.activations.relu)
    output  = dense1(input)
    dense2 =tf.keras.layers.Dense(3, activation= tf.keras.activations.softmax)(dense1)
    output  = dense2(dense1)

    model = tf.keras.Model(input.danse2)


    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])  #

    model.fit(x_train, y_train, epochs=1, verbose=0)
    print('acc:', model.evaluate(x_test, y_test, verbose=0))

    print(type(dense1))
    print(dense1, output)

    # print(tf.keras.backend.get_value(dense1,output))

    # new_model = tf.keras.Model(dense1.input, output)
    # print(new_model.predict(x_test))
    show_inside(dense1)
def linnerud_functional():

    def show_difference_all(preds, labels):
        diff = preds - labels
        error = np.mean(np.abs(diff), axis=0)
        print('평균 오차 :', error)

    x, y = datasets.load_linnerud(return_X_y=True)

    y0, y1, y2 = y[:,:1], y[:, 1:2], y[:, 2:]

    # 1번
    input = tf.keras.layers.Input([3])

    output0 = tf.keras.layers.Dense(5)(input)
    output0 = tf.keras.layers.Dense(1)(output0)

    output1 = tf.keras.layers.Dense(5)(input)
    output1 = tf.keras.layers.Dense(1)(output1)

    output2 = tf.keras.layers.Dense(5)(input)
    output2 = tf.keras.layers.Dense(3)(output2)
    output2 = tf.keras.layers.Dense(1)(output2)

    model = tf.keras.Model(input,[output0, output1, output2])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                  loss=tf.keras.losses.mae,
                  metrics = ['mse'])

    history = model.fit(x, [y0, y1, y2], epochs = 10, verbose=0)
    print('acc:', model.evaluate(x,[y0,y1,y2], verbose=0))
    print(history.history.keys())

    preds = model.predict(x, verbose=0)
    print(preds)
    print(type(preds), len(preds))

    show_difference_all(np.array(preds), y) # preds가 리턴하는 값이 리스트

    print(np.array(preds).shape)        # (3, 20, 1)
    print(np.hstack(preds).shape)       # (20, 3)

    show_difference_all(preds[0], y0)
    show_difference_all(preds[1], y1)
    show_difference_all(preds[2], y2)


# preds = model.predict(x)
# print(preds)
# softmax_iris_1()
softmax_iris_2()
# linnerud_functional()

# 문제
# show_different_all 함수를 사용해서    를 구하세요.

# 2번
# 평균 오차 : [53.15948658]
# 평균 오차 : [73.18978825]
# 평균 오차 : [21.45948105]

