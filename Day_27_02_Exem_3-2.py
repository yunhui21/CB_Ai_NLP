# Day_27_02_Exem_3-2.py

import os
import shutil
import urllib.request
import zipfile
import tensorflow as tf
import numpy as np

# 문제
# imagegenerator를 사용해서 test데이터에 대해 90%의 정확도를 구현하세요.
# 입력 shape을 300x300으로 사용하세요.

# url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip'
# urllib.request.urlretrieve(url, 'horse-or-human/horse-or-human.zip')

# zip_ref = zipfile.ZipFile('horse-or-human/horse-or-human.zip', 'r')
# zip_ref.extractall('horse-or-human/horse-or-human')
# zip_ref.close()

def model_4():
    conv_base = tf.keras.applications.VGG16(include_top=False, input_shape=[300, 300, 3])
    conv_base.trainable = False

    model = tf.keras.Sequential()
    model.add(conv_base)

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))                     # 오버피팅 방지
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=tf.keras.optimizers.RMSprop(0.0001),
                  loss=tf.keras.losses.binary_crossentropy,
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
    train_flow = train_gen.flow_from_directory('horse-or-human/horse-or-human',
                                               target_size=[300, 300],
                                               batch_size=batch_size,
                                               class_mode='binary')

    model.fit_generator(train_flow,
                                  steps_per_epoch=(500 + 527) // batch_size,
                                  epochs=1,
                                  # validation_data=train_flow,
                                  # validation_steps=(500 + 527) // batch_size,
                                  verbose=1)
model_4()
