# Day_35_02_KerasTuner.py
import tensorflow as tf
import kerastuner as kt
import numpy as np
from sklearn import preprocessing


def model_builder(hp):
    model = tf.keras.Sequential()

    hp_unit_1 = hp.Int('unit_1', min_value=256, max_value=512, step=256)
    model.add(tf.keras.layers.Dense(hp_unit_1, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(128, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax))

    hp_lr = hp.Choice('lr', values=[0.01, 0.001])
    model.compile(optimizer=tf.keras.optimizers.Adam(hp_lr),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])
    return model


# 12-3 파일에서 복사
def multi_layers_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)

    scaler = preprocessing.MinMaxScaler()
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    x_train = np.float32(x_train)
    x_test = np.float32(x_test)

    # ----------------------------------- #

    # tuner = kt.BayesianOptimization(model_builder,
    #                                 objective='val_acc',   # fit 결과로 만들어진 key 중의 하나
    #                                 max_trials=5,
    #                                 directory='keras_tuner/bayesian',
    #                                 project_name='mnist')
    # tuner = kt.RandomSearch(model_builder,
    #                         objective='val_acc',
    #                         max_trials=5,
    #                         directory='keras_tuner/random',
    #                         project_name='mnist')
    tuner = kt.Hyperband(model_builder,
                         objective='val_acc',
                         max_epochs=5,
                         directory='keras_tuner/hyperband',
                         project_name='mnist')

    # fit에 들어가는 매개변수를 그대로 전달
    tuner.search(x_train, y_train, epochs=2, verbose=2, batch_size=100, validation_split=0.2)

    # tuner.search_space_summary()
    # tuner.results_summary()

    best_hps = tuner.get_best_hyperparameters(num_trials=3)
    # print(best_hps)
    # print(best_hps[0])
    print(best_hps[0].get('lr'), best_hps[0].get('unit_1'))

    model = tuner.hypermodel.build(best_hps[0])
    model.fit(x_train, y_train, epochs=2, verbose=2, batch_size=100, validation_split=0.2)
    print('acc :', model.evaluate(x_test, y_test))


multi_layers_mnist()
