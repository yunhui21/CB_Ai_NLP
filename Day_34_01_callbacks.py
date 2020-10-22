# Day_34_01_callbacks.py
import tensorflow as tf
import pandas as pd
from sklearn import preprocessing, model_selection
import numpy as np
import tensorflow_hub as hub
import matplotlib.pyplot as plt

# 문제
# 콜백 중에서 plateau 단어가 들어간 것을 찾아서
# 3 patience에 대해 동작하도록 추가하세요

# 문제
# 에포크에 포함된 batch 데이터를 처리할 때마다 결과를 보고 싶습니다

# 문제
# 복사한 코드에 대해 파인 튜닝을 하세요 (에포크 2회)
# 만들어져 있는 test 제너레이터를 train으로 사용하세요
# 다시 말해, test 제너레이터에 대해 fit 함수를 한번만 호출하면 됩니다
# 예측 결과는 복사한 코드에 있는 걸 수정없이 사용합니다

# 문제
# batch에 대한
def get_cars_sparse():
    cars = pd.read_csv('data/car.data',
                       header=None,
                       names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'eval'])

    lb = preprocessing.LabelEncoder()

    buying = lb.fit_transform(cars.buying)
    maint = lb.fit_transform(cars.maint)
    doors = lb.fit_transform(cars.doors)
    persons = lb.fit_transform(cars.persons)
    lug_boot = lb.fit_transform(cars.lug_boot)
    safety = lb.fit_transform(cars.safety)
    eval = lb.fit_transform(cars['eval'])

    print(buying)

    # x = np.array([buying, maint, doors, persons, lug_boot, safety])       # (6, 1728)
    x = np.transpose([buying, maint, doors, persons, lug_boot, safety])     # (1728, 6)
    y = eval

    print(x.shape, y.shape)     # (1728, 6) (1728,)
    return np.float32(x), y  #, lb.classes_


def cars_evaluation_sparse_plateau():
    x, y = get_cars_sparse()

    indices = np.arange(len(x))
    np.random.shuffle(indices)

    x = x[indices]
    y = y[indices]

    test_size = int(len(x) * 0.2)
    train_size = len(x) - test_size * 2

    x_train, x_valid, x_test = x[:train_size], x[train_size:train_size+test_size], x[-test_size:]
    y_train, y_valid, y_test = y[:train_size], y[train_size:train_size+test_size], y[-test_size:]

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(len(set(y)), activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.SGD(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])

    plateau = tf.keras.callbacks.ReduceLROnPlateau(patience=3, verbose=1)

    model.fit(x_train, y_train, epochs=100, batch_size=32,
              validation_data=(x_valid, y_valid), verbose=2,
              callbacks=[plateau])

    print('acc :', model.evaluate(x_test, y_test, verbose=0))


def get_resnet50():
    # url = 'https://tfhub.dev/tensorflow/resnet_50/classification/1'
    url = 'https://tfhub.dev/tensorflow/resnet_50/feature_vector/1'

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input([224, 224, 3]))
    model.add(hub.KerasLayer(url, trainable=False))
    model.add(tf.keras.layers.Dense(5, activation='softmax'))

    return model

# class EveryBatch(tf.keras.callbacks.Callback):
#     def __int__(self):
#         self.batch_loss = []
#         self.batch_acc = []
#     def on_batch_begin(self, batch, logs=None):
#         print('on_batch_begin', logs)
#
#     def on_batch_end(self, batch, logs=None):
#         print('on_batch_end', logs) # dictionary를 생성
#
#         self.batch_loss.append(logs['loss'])
#         self.batch_acc.append(logs['acc'])
#         self.model.reset_metrics()

class EveryBatch(tf.keras.callbacks.Callback):
    def __init__(self):
        self.batch_loss = []
        self.batch_acc = []

    def on_batch_begin(self, batch, logs=None):
        print('on_batch_begin', logs)

    def on_batch_end(self, batch, logs=None):
        print('on_batch_end', logs)

        self.batch_loss.append(logs['loss'])
        self.batch_acc.append(logs['acc'])
        self.model.reset_metrics()


# 파인 튜닝을 해서 정확도를 대폭 상승시켰다
def simple_transfer_learning():
    images_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
    images_path = tf.keras.utils.get_file('flower_photos', images_url, untar=True)
    # print(images_path)

    test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)

    test_flow = test_gen.flow_from_directory(images_path,
                                             target_size=[224, 224],
                                             batch_size=32,
                                             class_mode='sparse')
    # x, y = test_flow.next()
    x, y = next(test_flow)
    print(x.shape, y.shape)     # (32, 224, 224, 3) (32,)

    model = get_resnet50()

    # steps_per_epoch 계산
    print(test_flow.samples, test_flow.batch_size) # 3670 32

    every_batch =EveryBatch()

    steps_per_epoch = test_flow.samples // test_flow.batch_size

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])

    model.fit(test_flow, epochs=1, steps_per_epoch=steps_per_epoch,
              callbacks=[every_batch])

    indices = range(len(every_batch.batch_loss))

    plt.subplot(1, 2, 1)
    plt.plot(indices, every_batch.batch_loss)
    plt.title('loss')

    plt.subplot(1, 2, 2)
    plt.plot(indices, every_batch.batch_acc)
    plt.title('acc')

    plt.show()

    # preds = model.predict(x, verbose=0)
    #
    # labels = {v:k for k, v in test_flow.class_indices.items()}
    #
    # labels = [name for _, name in sorted(labels.items())]
    #
    # preds_arg = np.argmax(preds, axis=-1)
    #
    # plt.figure(figsize=(12, 6))
    # for i, (img, label, pred) in enumerate(zip(x, y, preds_arg)):
    #     plt.subplot(4, 8, i+1)
    #     # plt.title('{}: {}'.format(labels[int(label)], labels[pred]))
    #     plt.title(labels[pred], color='g' if int(label) == pred else 'r')
    #     plt.axis('off')
    #     plt.imshow(img)
    #
    # plt.show()

# cars_evaluation_sparse_plateau()

simple_transfer_learning()
