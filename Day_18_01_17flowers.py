# Day_18_01_17flowers.py

import tensorflow as tf
import numpy as np
from PIL import Image # pillow
# import PIL 자동완성기능이 안된다.
import matplotlib as plt
import os
from sklearn import model_selection



# 문제
# origin 폴더에 있는 이미지를 224호 축소해서 224 폴더로 복사하세요.
# 이름을 17flowers_origin


# 문제
# 꽃 폴더로부터 파일을 읽어서 x, y 데이터를 만드세요.
# y: 80개 간격이 나누어져 있습니다. (0-16)
# x: 4차원 배열

# 문제
# 꽃 데이터를 20%에 대해서 결과를 예측하세요.


# 문제
# 이미지 데이터를 스케일링 하세요.

# 문제
# 정확도를 50%정도 올려보세요.


def make_flowers_224(src_folder, dst_folder, img_size=[224, 224]):
    for filename in os.listdir(src_folder):
        print(filename)

        items = filename.split('.')
        if items[-1] != 'jpg':
            continue

        # file_path = src_folder + '/' + filename
        # print(file_path)
        file_path = os.path.join(src_folder, filename)
        print(file_path)
        image1 = Image.open(file_path)
        image2 = image1.resize(img_size)
        image2.save(os.path.join(dst_folder, filename))


def make_flowers_xy(dir_name):
    x , y = [], []
    for filename in os.listdir(dir_name):
        items = filename.split('.')
        if items[-1] != 'jpg':
            continue

        idx = int(items[0].split('_')[1])
        print(idx, (idx-1) // 80)
        y.append((idx - 1)// 80)

        file_path = os.path.join(dir_name, filename)
        img = Image.open(file_path)
        # print(type(img))
        # print(type(np.array(img)), np.array(img).shape)

        np.img = np.array(img)
        x.append(np.img)

    return np.float32(x), np.float32(y), len(set(y))


    # make_flowers_224('data/17flowers_origin', 'data/17flowers_224', img_size=[56, 56])
    # make_flowers_224('data/17flowers_origin', 'data/17flowers_224', img_size=[224, 224])
def model_flowers_dense():
    x, y, n_classes = make_flowers_xy('data/17flowers_224')
    # print(np.min(x), np.max(x)) # 0.0 255.0

    x = x/255

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.8)

    model = tf.keras.Sequential()
    # model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=[3,3],
    #                                  strides=[1,1], padding='same',
    #                                  activation= tf.keras.activations.relu,
    #                                  input_shape=[56, 56, 1]))

    model.add(tf.keras.layers.Conv2D(32, [3,3],[1,1],'same',activation= tf.keras.activations.relu,input_shape=x[0].shape))
    model.add(tf.keras.layers.MaxPool2D([2,2], [2,2], 'same'))

    model.add(tf.keras.layers.Conv2D(64, [3,3],[1,1],'same',activation= tf.keras.activations.relu))
    model.add(tf.keras.layers.MaxPool2D([2,2], [2,2], 'same'))

    model.add(tf.keras.layers.Conv2D(128, [3,3],[1,1],'same',activation= tf.keras.activations.relu))
    model.add(tf.keras.layers.MaxPool2D([2,2], [2,2], 'same'))

    model.add(tf.keras.layers.Conv2D(128, [3,3],[1,1],'same',activation= tf.keras.activations.relu))
    model.add(tf.keras.layers.MaxPool2D([2,2], [2,2], 'same'))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(1024, tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(256, tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(n_classes, tf.keras.activations.softmax))

    model.summary()
    # exit(-1)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss = tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])

    model.fit(x_train, y_train, epochs=50, batch_size= 100, verbose=2, validation_split=0.2)
    print(model.evaluate(x_test, y_test, verbose=2))


def model_flowers_1x1_conv():
    x, y, n_classes = make_flowers_xy('data/17flowers_224')
    # print(np.min(x), np.max(x)) # 0.0 255.0

    x = x / 255

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.8)
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(32, [3, 3], [1, 1], 'same', activation=tf.keras.activations.relu, input_shape=x[0].shape))
    model.add(tf.keras.layers.MaxPool2D([2, 2], [2, 2], 'same'))

    model.add(tf.keras.layers.Conv2D(64, [3, 3], [1, 1], 'same', activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.MaxPool2D([2, 2], [2, 2], 'same'))

    model.add(tf.keras.layers.Conv2D(128, [3, 3], [1, 1], 'same', activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.MaxPool2D([2, 2], [2, 2], 'same'))

    model.add(tf.keras.layers.Conv2D(128, [3, 3], [1, 1], 'same', activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.MaxPool2D([2, 2], [2, 2], 'same'))

    model.add(tf.keras.layers.Conv2D(1024,[7,7],[1,1],'valid', activation= tf.keras.activations.relu))
    model.add(tf.keras.layers.Conv2D(256,[1,1], [1,1], activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Conv2D(n_classes,[1,1], [1,1], activation=None))
    model.add(tf.keras.layers.Reshape([n_classes]))
    model.add(tf.keras.layers.Softmax())

    model.summary()

    # exit(-1)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])

    model.fit(x_train, y_train, epochs=50, batch_size=100, verbose=2, validation_split=0.2)
    print(model.evaluate(x_test, y_test, verbose=2))

# model_flowers_dense()
model_flowers_1x1_conv()