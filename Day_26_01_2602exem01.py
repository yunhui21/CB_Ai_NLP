# Day_26_01_2602exem01.py
import tensorflow as tf
from sklearn import feature_extraction, model_selection, linear_model
import numpy as np
# 아래 데이터로 학습해서 새로운 입력이 들어왔을 때의 결과를 예측하세요.
# x_test =
# mae를 사용하고 , 오차는 0.001아래로 받으세요,
# 2.0버전에서도 에러가 나지 않아야 합니다.
def save_model():
    xs = [0,1,2,3,4,5,6]
    ys = [-3,-2,-1,0,1,2,3]

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1, input_dim=1))
    # model.add(tf.keras.layers.Input([1]))
    # model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer='sgd', loss='mse')
    # optimizer를 바꾸고, adam lr조정필요
    model.fit(xs, ys, epochs=10, verbose=0)
    print(model.evaluate(xs, ys))

    model.save('data/mymodel_1.h5')


def load_model():
    model = tf.keras.models.load_model('data/mymodel_1.h5')

    x_test = [20, 30,39, 40, 60]
    y_test = [17, 27, 36, 37, 57]

    preds = model.predict(x_test)
    print(preds)

    diff = y_test - preds.reshape(-1)
    diff_abs = np.abs(diff)
    print('mae:', np.mean(diff_abs))


save_model()
load_model()

