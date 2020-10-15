# Day_29_01_17flowers.py

import tensorflow as tf
import numpy as np
import shutil
from PIL import Image # pillow
# import PIL 자동완성기능이 안된다.
import matplotlib as plt
import os
from sklearn import model_selection, preprocessing

# 문제
# 17flowers에 대해서 80%로 학습하고 20%에 대해서 정확도를 예측하세요.
# 정확도를 80%로 목표로 합니다.

# 원하는 사이즈로 변환해서 사용


def make_flowers_xy(dir_name, img_size):
    x , y = [], []
    for filename in os.listdir(dir_name):
        items = filename.split('.')
        if items[-1] != 'jpg':
            continue

        idx = int(items[0].split('_')[1])
        print(idx, (idx-1) // 80)
        y.append((idx - 1)// 80)

        file_path = os.path.join(dir_name, filename)

        img1 = Image.open(file_path)
        img2 = img1.resize([img_size, img_size])
        # print(type(img))
        # print(type(np.array(img)), np.array(img).shape)

        np.img = np.array(img2)
        x.append(np.img)

    return np.float32(x), np.float32(y), len(set(y))


    # make_flowers_224('data/17flowers_origin', 'data/17flowers_224', img_size=[56, 56])
    # make_flowers_224('data/17flowers_origin', 'data/17flowers_224', img_size=[224, 224])

def model_flowers_01():
    img_size = 112
    x, y, n_classes = make_flowers_xy('17flowers_origin', img_size=img_size)
    # print(np.min(x), np.max(x)) # 0.0 255.0

    x = x/255

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.8)

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Input(x[0].shape))
    model.add(tf.keras.layers.Conv2D(32, [3,3],[1,1],'same',activation= 'relu'))
    model.add(tf.keras.layers.MaxPool2D([2,2], [2,2], 'same'))

    model.add(tf.keras.layers.Conv2D(64, [3,3],[1,1],'same',activation= 'relu'))
    model.add(tf.keras.layers.MaxPool2D([2,2], [2,2], 'same'))

    model.add(tf.keras.layers.Conv2D(128, [3,3],[1,1],'same',activation= 'relu'))
    model.add(tf.keras.layers.MaxPool2D([2,2], [2,2], 'same'))

    model.add(tf.keras.layers.Conv2D(128, [3,3],[1,1],'same',activation= 'relu'))
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

    model.fit(x_train, y_train, epochs=50, batch_size= 100, verbose=2, validation_data=[x_test, y_test])
    # print(model.evaluate(x_test, y_test, verbose=2))

# 사전학습
def model_flowers_02():
    img_size = 112
    x, y, n_classes = make_flowers_xy('17flowers_origin', img_size=img_size)
    # print(np.min(x), np.max(x)) # 0.0 255.0

    x = x/255

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.8)

    conv_base = tf.keras.applications.VGG16(include_top=False)
    conv_base.trainable = False
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(x[0].shape))
    model.add(conv_base)

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(512, tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(n_classes, tf.keras.activations.softmax))

    model.summary()
    # exit(-1)

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss = tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])

    model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=1, validation_data=[x_test, y_test])
    # print(model.evaluate(x_test, y_test, verbose=2))

# 이미지증식
# 케라스 이미지 증식에는 전제조건이 필요

# 문제 1
# 17flowers 폴더의 내용ㅇ르 케라스 이밎 증감을 할 수 있는 형태로 변환하세요.
# 새 폴더 : 17flowers, 17flowers/train, 17flowers/valid

def make_augmentation_folders():
    def make_folder(path):
        if not os.path.exists(path):
            os.mkdir(path)

    make_folder('17flowers')
    make_folder('17flowers/train')
    make_folder('17flowers/valid')

    for filename in os.listdir('17flowers_origin'):
        items = filename.split('.')
        if items[-1] != 'jpg':
            continue

        idx = int(items[0].split('_')[1])
        idx -= 1
        label = idx // 80

        if idx % 80 == 0:
            make_folder('17flowers/train/{}'.format(label))
        elif idx % 80 == 63:
            make_folder('17flowers/valid/{}'.format(label))

        src_path = os.path.join('17flowers_origin', filename)
        dst_path = '17flowers/{}/{}/{}'.format('train' if idx % 80 <= 63 else 'valid', label, filename)

        shutil.copy(src_path, dst_path)

# 문제
# 이미지 증감 코드를 사용해서 정확도를 예측하세요.

def model_flowers_03():
    img_size = 112
    train_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255,
                                                                rotation_range=20,
                                                                width_shift_range=0.1,
                                                                height_shift_range=0.1,
                                                                shear_range=0.1,
                                                                zoom_range=0.1,
                                                                horizontal_flip=True)
    valid_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)

    batch_size = 32
    train_flow = train_gen.flow_from_directory('17flowers/train',
                                               target_size=[img_size, img_size],
                                               batch_size=batch_size,
                                               class_mode='sparse')
    valid_flow = valid_gen.flow_from_directory('17flowers/valid',
                                               target_size=[img_size, img_size],
                                               batch_size=batch_size,
                                               class_mode='sparse')

    conv_base = tf.keras.applications.VGG16(include_top=False)
    conv_base.trainable = False

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input([img_size, img_size, 3]))
    model.add(conv_base)

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(512, tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(17, tf.keras.activations.softmax))

    model.summary()
    # exit(-1)

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss = tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])

    model.fit_generator(train_flow, epochs=10, validation_data=valid_flow)

# 문제
# imagedatagenerator 의 flow를 사용해서 정확도를  예측하세요.
def model_flowers_04():
    img_size = 112
    x, y, n_classes = make_flowers_xy('17flowers_origin', img_size=img_size)
     = flow(x, y=None, batch_size=32, shuffle=True, sample_weight=None, seed=None,
        save_to_dir=None, save_prefix='', save_format='', subset=None    )

    x = x / 255

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.8)

    conv_base = tf.keras.applications.VGG16(include_top=False)
    conv_base.trainable = False
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(x[0].shape))
    model.add(conv_base)

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(512, tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(n_classes, tf.keras.activations.softmax))

    model.summary()
    # exit(-1)

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])

    model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=1, validation_data=[x_test, y_test])
# model_flowers_01()
# model_flowers_02()
# make_augmentation_folders()
# generator_basic()
# model_flowers_03()
model_flowers_04()