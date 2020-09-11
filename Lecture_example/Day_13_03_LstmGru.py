# Day_13_03_LstmGru.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def simple_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.SimpleRNN(30, return_sequences=True,
                                        input_shape=[100, 2]))
    model.add(tf.keras.layers.SimpleRNN(30))
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mse)
    model.summary()

    return model


def lstm_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(30, return_sequences=True, input_shape=[100, 2]))
    model.add(tf.keras.layers.LSTM(30))
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mse)
    model.summary()

    return model


def gru_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.GRU(30, return_sequences=True, input_shape=[100, 2]))
    model.add(tf.keras.layers.GRU(30))
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mse)
    model.summary()

    return model


def show_result(history):
    plt.plot(history.history['loss'], 'b--', label='loss')
    plt.plot(history.history['val_loss'], 'r--', label='val_loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()


def show_model(model_func):
    size = 100
    np.random.seed(41)

    x, y = [], []
    for _ in range(3000):
        values = np.random.rand(size)
        i0, i1 = np.random.choice(size, 2, replace=False)

        two_hots = np.zeros(size)
        two_hots[[i0, i1]] = 1
        # [0 0 0 ... 0 1 0 .. 0 0 1 0 0]

        xx = np.transpose([two_hots, values])   # (100, 2)
        yy = values[i0] * values[i1]    # sum(two_hots * values)

        x.append(xx)        # (3000, 100, 2)
        y.append(yy)        # (3000,)

    x = np.float32(x)
    y = np.float32(y)
    print(x.shape, y.shape)     # (3000, 100, 2) (3000,)

    x_train, x_test = x[:2560], x[2560:]
    y_train, y_test = y[:2560], y[2560:]

    model = model_func()

    history = model.fit(x_train, y_train, epochs=2, verbose=2,
                        validation_split=0.2, batch_size=32)
    # show_result(history)

    # print(type(history))            # <class 'tensorflow.python.keras.callbacks.History'>
    # print(type(history.history))    # <class 'dict'>
    #
    # print(history.history.keys())   # dict_keys(['loss', 'val_loss'])
    # print(history.history)
    # {'loss': [0.05972842552000657, 0.049569002498174086],
    # 'val_loss': [0.04432915383949876, 0.044075330486521125]}

    # print(model.evaluate(x_test, y_test))
    #
    # preds = model.predict(x_test)
    # # print(preds.shape)            # (440, 1)
    #
    # preds = preds.reshape(-1)       # (440,)
    # errors = np.abs(preds - y_test)
    #
    # print('acc :', np.mean(errors <= 0.04))


# show_model(simple_model)
# show_model(lstm_model)
show_model(gru_model)

# simple
# 0.051354074545881964
# acc : 0.075

# lstm
# 0.0006232027136403221
# acc : 0.9136363636363637

# gru
# 0.00028069077736952086
# acc : 0.9840909090909091

# (100, 784) -> (784, 256) -> (256, 128) -> (128, 10)
