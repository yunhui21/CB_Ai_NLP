# Day_26_01_Exem_3_1.py
import os
import shutil
import urllib.request
import zipfile
import tensorflow as tf
import numpy as np

# rps = rack + paper + scissor image data
# url = 'https://storage.googleapis.com/download.tensorflow.org/data/rps.zip'
# urllib.request.urlretrieve(url, 'rps/rps.zip')

# zip_ref = zipfile.ZipFile('rps/rps.zip', 'r')
# zip_ref.extractall('rps/rps')
# zip_ref.close()


# url = 'https://storage.googleapis.com/download.tensorflow.org/data/rps-test-set.zip'
# urllib.request.urlretrieve(url, 'rps/rps-test-set.zip')
#
# zip_ref = zipfile.ZipFile('rps/rps-test-set.zip', 'r')
# zip_ref.extractall('rps/rps')
# zip_ref.close()

# 문제
# imagegenerator를 사용해서 test데이터에 대해 90%의 정확도를 구현하세요.
# 입력 shape을 150x150으로 변환하세요.

def model_4():
    conv_base = tf.keras.applications.VGG16(include_top=False, input_shape=[150, 150, 3])
    conv_base.trainable = False

    model = tf.keras.Sequential()
    model.add(conv_base)

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))                     # 오버피팅 방지
    model.add(tf.keras.layers.Dense(3, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.RMSprop(0.0001),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])

    train_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255,
                                                                rotation_range=20,
                                                                width_shift_range=0.1,
                                                                height_shift_range=0.1,
                                                                shear_range=0.1,
                                                                zoom_range=0.1,
                                                                horizontal_flip=True)
    valid_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

    batch_size = 32
    # onthot/sparse
    train_flow = train_gen.flow_from_directory('rps/rps',
                                               target_size=[150, 150],
                                               batch_size=batch_size,
                                               class_mode='sparse')
    valid_flow = valid_gen.flow_from_directory('rps/rps-test-set',
                                               target_size=[150, 150],
                                               batch_size=batch_size,
                                               class_mode='sparse')
    model.fit_generator(train_flow,
                                  steps_per_epoch=(840*3) // batch_size,
                                  epochs=1,
                                  validation_data=valid_flow,
                                  validation_steps=batch_size,
                                  verbose=1)
model_4()

