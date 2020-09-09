# Day_13_03_lstmGru.py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def simple_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.SimpleRNN(30, return_sequences=True, input_shape=[100, 2]))
    model.add(tf.keras.layers.SimpleRNN(30))        # 3차원을 받아야 함.
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.mse)
    model.summary()

    return model

def lstm_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(30, return_sequences=True, input_shape=[100, 2]))
    model.add(tf.keras.layers.LSTM(30))        # 3차원을 받아야 함.
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.mse)
    model.summary()

    return model

def gru_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.GRU(30, return_sequences=True, input_shape=[100, 2]))
    model.add(tf.keras.layers.GRU(30))        # 3차원을 받아야 함.
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.mse)
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
        # print(values)
        # print(values.shape)
        # break
        i0, i1 = np.random.choice(size, 2, replace = False)
        # print(i0, i1)
        # break

        two_hots = np.zeros(size)   # 0이 100개
        two_hots[[i0, i1]] = 1      # 2개만 1

        # [0, 0, 0, ...,1,0 ,0 , 1,0 .0]
        xx = np.transpose([two_hots, values])
        # print(xx)
        # break
        yy = values[i0] * values[i1]    # sum(two_hots * values)

        x.append(xx)    #  (3000, 100, 2)
        y.append(yy)    # (3000,
    x = np.float32(x)
    y = np.float32(y)
    print(x.shape, y.shape)     # (3000, 100, 2) (3000,)

    x_train, x_test = x[:2560], x[2560:]
    y_train, y_test = y[:2560], y[2560:]

    model = model_func()

    history = model.fit(x_train, y_train, epochs = 2, verbose=2,
                        validation_split=0.2, batch_size = 32)
    # show_result(history)
    # print(type(history))        #<class 'tensorflow.python.keras.callbacks.History'>
    # print(type(history.history))#<class 'dict'>

    # print(history.history.keys())# dict_keys(['loss', 'val_loss'])
    # print(history.history)# {'loss': [0.07009455258958042, 0.04938422949635424], 'val_loss': [0.04431653651408851, 0.044186891871504486]}

    # print(model.evaluate(x_test, y_test))
    #
    # preds = model.predict(x_test)
    # print(preds.shape)
    #
    # preds = preds.reshape(-1)
    # erros = np.abs(preds - y_test)
    #
    # print('acc:', np.mean(erros <= 0.04))

# show_model(simple_model)
show_model(lstm_model)
show_model(gru_model)

# lstm
# 0.0016438316998325965
# acc: 0.7136363636363636
# gru
# 0.00012521337157522794
# acc: 0.9795454545454545

