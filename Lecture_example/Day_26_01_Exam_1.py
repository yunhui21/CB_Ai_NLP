# Day_26_01_Exam_1.py

# 문제
# 아래 데이터로 학습해서 새로운 입력이 들어왔을 때의 결과를 예측하세요
# x_test = [25, 30, 35, 40, 45]
# 결과는 mae를 사용하고, 오차는 0.001 아래로 만드세요
# 2.0 버전에서도 에러가 나지 않아야 합니다
import tensorflow as tf
import numpy as np


def save_model():
    xs = [0, 1, 2, 3, 4, 5, 6]
    ys = [-3, -2, -1, 0, 1, 2, 3]

    model = tf.keras.Sequential()
    # model.add(tf.keras.layers.Dense(1, input_dim=1))

    model.add(tf.keras.layers.Input([1]))
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer='sgd', loss='mse')

    model.fit(xs, ys, epochs=10000, verbose=0)
    print(model.evaluate(xs, ys))

    model.save('data/mymodel_1.h5')


def load_model():
    model = tf.keras.models.load_model('data/mymodel_1.h5')

    x_test = [25, 30, 35, 40, 45]
    y_test = [22, 27, 32, 37, 42]
    preds = model.predict(x_test)
    print(preds)

    diff = y_test - preds.reshape(-1)
    diff_abs = np.abs(diff)
    print('mae :', np.mean(diff_abs))   # mae : 0.000145721435546875


save_model()
load_model()


