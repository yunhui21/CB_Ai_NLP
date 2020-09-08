# Day_11_01_KerasBasic.py
import tensorflow as tf
import csv
import numpy as np

# 문제
# 속도가 30과 50일 때의 제동 거리를 구하세요
# (csv 파일을 읽을 때, csv 모듈을 사용해서 읽어봅니다)

# 문제
# girth가 10이고 height가 70일 때와
# girth가 20이고 height가 80일 때의 volume을 예측하세요


def linear_basic():
    x = [1, 2, 3]
    y = [1, 2, 3]

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1, input_dim=1))

    model.compile(optimizer='sgd', loss='mse')

    model.fit(x, y, epochs=100)
    print(model.evaluate(x, y))

    preds = model.predict(x)
    print(preds)

    model.summary()


def linear_cars():
    x, y = [], []

    f = open('data/cars.csv', 'r', encoding='utf-8')
    f.readline()

    for row in csv.reader(f):
        # print(row)
        x.append(int(row[1]))
        y.append(int(row[2]))
    f.close()

    # ------------------------------ #

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1, input_dim=1))

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001),
                  loss=tf.keras.losses.mse)

    model.fit(x, y, epochs=100, verbose=2)
    print(model.evaluate(x, y))

    preds = model.predict([30, 50])
    print(preds)


def multiple_trees():
    x, y = [], []

    f = open('data/trees.csv', 'r', encoding='utf-8')
    f.readline()

    for row in csv.reader(f):
        # print(row)
        x.append([float(row[1]), float(row[2])])
        y.append(float(row[-1]))
    f.close()

    # y = y.reshape(-1, 1)        # error
    y = np.reshape(y, [-1, 1])
    x = np.float32(x)

    print(np.array(x).shape)
    print(np.array(y).shape)

    # ------------------------------ #

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001),
                  loss=tf.keras.losses.mse)

    model.fit(x, y, epochs=100, verbose=2)
    print(model.evaluate(x, y))

    preds = model.predict(np.float32([[10, 70],
                                      [20, 80]]))
    print(preds)


# linear_basic()
# linear_cars()
multiple_trees()
