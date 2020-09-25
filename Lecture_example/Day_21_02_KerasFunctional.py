# Day_21_02_KerasFunctional.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 문제 1
# AND 연산에 대한 데이터를 생성해서 결과를 예측하세요

# 문제 2
# XOR 연산에 대한 데이터를 생성해서 결과를 예측하세요


def regression_and():
    data = [[0, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 1]]
    data = np.float32(data)

    x = data[:, :-1]
    y = data[:, -1:]

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid))

    model.compile(optimizer='sgd',
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['acc'])

    model.fit(x, y, epochs=1000, verbose=2)
    print('acc :', model.evaluate(x, y))

    preds = model.predict(x)
    print(preds)


def regression_xor():
    data = [[0, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]]
    data = np.float32(data)

    x = data[:, :-1]
    y = data[:, -1:]

    model = tf.keras.Sequential()
    model.add(tf.keras.Input([2]))
    model.add(tf.keras.layers.Dense(2, activation=tf.keras.activations.sigmoid))
    model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid))

    model.compile(optimizer=tf.keras.optimizers.Adam(0.1),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['acc'])

    model.fit(x, y, epochs=1000, verbose=2)
    print('acc :', model.evaluate(x, y))

    preds = model.predict(x)
    print(preds)


# 기본 버전
def regression_xor_functional_1():
    data = [[0, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]]
    data = np.float32(data)

    x = data[:, :-1]
    y = data[:, -1:]

    # model = tf.keras.Sequential()
    # model.add(tf.keras.Input([2]))
    # model.add(tf.keras.layers.Dense(2, activation=tf.keras.activations.sigmoid))
    # model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid))

    # 1번
    # input = tf.keras.Input([2])
    # dense1 = tf.keras.layers.Dense(2, activation=tf.keras.activations.sigmoid)
    # output1 = dense1.__call__(input)
    # dense2 = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)
    # output2 = dense2.__call__(output1)

    # # 2번
    # input = tf.keras.Input([2])
    # dense1 = tf.keras.layers.Dense(2, activation=tf.keras.activations.sigmoid)
    # output1 = dense1(input)
    # dense2 = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)
    # output2 = dense2(output1)

    # 3번
    input = tf.keras.Input([2])
    output1 = tf.keras.layers.Dense(2, activation=tf.keras.activations.sigmoid)(input)
    output2 = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)(output1)

    model = tf.keras.Model(input, output2)

    model.compile(optimizer=tf.keras.optimizers.Adam(0.1),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['acc'])

    model.fit(x, y, epochs=1000, verbose=2)
    print('acc :', model.evaluate(x, y))

    preds = model.predict(x)
    print(preds)


# 멀티 입력 버전
def regression_xor_functional_2():
    data = [[0, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]]
    data = np.float32(data)

    x1 = data[:, :1]
    x2 = data[:, 1:2]
    y = data[:, 2:]

    # 1번
    # input_left = tf.keras.Input([1])
    # input_rite = tf.keras.Input([1])
    # output1 = tf.keras.layers.concatenate([input_left, input_rite], axis=1)

    # 2번
    input_left = tf.keras.Input([1])
    output_left = tf.keras.layers.Dense(2, activation=tf.keras.activations.sigmoid)(input_left)

    input_rite = tf.keras.Input([1])
    output_rite = tf.keras.layers.Dense(2, activation=tf.keras.activations.sigmoid)(input_rite)

    output1 = tf.keras.layers.concatenate([output_left, output_rite], axis=1)

    output2 = tf.keras.layers.Dense(20, activation=tf.keras.activations.sigmoid)(output1)
    output3 = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)(output2)

    model = tf.keras.Model([input_left, input_rite], output3)

    model.compile(optimizer=tf.keras.optimizers.Adam(0.1),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['acc'])

    model.fit([x1, x2], y, epochs=1000, verbose=2)
    print('acc :', model.evaluate([x1, x2], y))

    preds = model.predict([x1, x2])
    print(preds)


# 멀티 입출력 버전
def regression_xor_functional_3():
    data = [[0, 0, 0, 0],
            [0, 1, 1, 0],
            [1, 0, 1, 0],
            [1, 1, 0, 1]]
    data = np.float32(data)

    x1 = data[:, :1]
    x2 = data[:, 1:2]
    y1 = data[:, 2:3]
    y2 = data[:, 3:]

    input_left = tf.keras.Input([1])
    output_left = tf.keras.layers.Dense(2, activation=tf.keras.activations.sigmoid)(input_left)

    input_rite = tf.keras.Input([1])
    output_rite = tf.keras.layers.Dense(2, activation=tf.keras.activations.sigmoid)(input_rite)

    output = tf.keras.layers.concatenate([output_left, output_rite], axis=1)

    output_left_1 = tf.keras.layers.Dense(2, activation=tf.keras.activations.sigmoid)(output)
    output_left_2 = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)(output_left_1)

    output_rite_1 = tf.keras.layers.Dense(2, activation=tf.keras.activations.sigmoid)(output)
    output_rite_2 = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)(output_rite_1)

    model = tf.keras.Model([input_left, input_rite], [output_left_2, output_rite_2])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.1),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['acc'])

    history = model.fit([x1, x2], [y1, y2], epochs=1000, verbose=2)
    print(model.evaluate([x1, x2], [y1, y2], verbose=0))
    # [0.0017626925837248564, 0.0008243226, 0.00093837007, 1.0, 1.0]

    # print(model.predict([x1, x2]))
    # print(history.history)

    # model.summary()
    print(history.history.keys())
    # dict_keys(['loss', 'dense_3_loss', 'dense_5_loss', 'dense_3_acc', 'dense_5_acc'])

    # 문제
    # history 객체를 시각화하세요 (loss와 accuracy)
    epochs = np.arange(1000) + 1

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['dense_3_loss'], 'r', label='xor')
    plt.plot(epochs, history.history['dense_5_loss'], 'g', label='and')
    plt.title('loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['dense_3_acc'], 'r', label='xor')
    plt.plot(epochs, history.history['dense_5_acc'], 'g', label='and')
    plt.title('accuracy')
    plt.legend()

    plt.show()




# regression_and()
# regression_xor()
# regression_xor_functional_1()
# regression_xor_functional_2()
regression_xor_functional_3()
