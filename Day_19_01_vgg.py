# Day_19_01_vgg.py
import tensorflow as tf
import numpy as np
import os
from sklearn import model_selection
from PIL import Image



# 문제
# vgg16 모델을 구축하세요.
# 학습없이 summenry 함수로 결과를 확인하세요.

# 문제
# 앞에서 만든 모델의 fc레이어들을 컨볼루션 레이어로 교체하세요.
def vgg_16_dense():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input([224,224,3]))
    model.add(tf.keras.layers.Conv2D(64, [3,3], [1,1], 'same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(64, [3,3], [1,1], 'same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D([2,2], [2,2], 'same'))

    model.add(tf.keras.layers.Conv2D(128, [3, 3], [1, 1], 'same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(128, [3, 3], [1, 1], 'same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D([2,2], [2,2], 'same')) #

    model.add(tf.keras.layers.Conv2D(256, [3, 3], [1, 1], 'same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(256, [3, 3], [1, 1], 'same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(256, [3, 3], [1, 1], 'same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D([2,2], [2,2], 'same')) #

    model.add(tf.keras.layers.Conv2D(512, [3, 3], [1, 1], 'same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(512, [3, 3], [1, 1], 'same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(512, [3, 3], [1, 1], 'same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D([2,2], [2,2], 'same')) #

    model.add(tf.keras.layers.Conv2D(512, [3, 3], [1, 1], 'same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(512, [3, 3], [1, 1], 'same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(512, [3, 3], [1, 1], 'same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D([2,2], [2,2], 'same')) #

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(4096, 'relu'))
    model.add(tf.keras.layers.Dense(4096, 'relu'))
    model.add(tf.keras.layers.Dense(1000, 'softmax'))

    model.summary()
    return

def vgg_16_txt_conv():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input([224, 224, 3]))
    model.add(tf.keras.layers.Conv2D(64, [3, 3], [1, 1], 'same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(64, [3, 3], [1, 1], 'same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D([2, 2], [2, 2], 'same'))

    model.add(tf.keras.layers.Conv2D(128, [3, 3], [1, 1], 'same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(128, [3, 3], [1, 1], 'same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D([2, 2], [2, 2], 'same'))  #

    model.add(tf.keras.layers.Conv2D(256, [3, 3], [1, 1], 'same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(256, [3, 3], [1, 1], 'same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(256, [3, 3], [1, 1], 'same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D([2, 2], [2, 2], 'same'))  #

    model.add(tf.keras.layers.Conv2D(512, [3, 3], [1, 1], 'same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(512, [3, 3], [1, 1], 'same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(512, [3, 3], [1, 1], 'same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D([2, 2], [2, 2], 'same'))  #

    model.add(tf.keras.layers.Conv2D(512, [3, 3], [1, 1], 'same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(512, [3, 3], [1, 1], 'same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(512, [3, 3], [1, 1], 'same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D([2, 2], [2, 2], 'same'))  #

    model.add(tf.keras.layers.Conv2D(4096, [7,7],[1,1],'valid', activation='relu'))
    model.add(tf.keras.layers.Conv2D(4096, [1,1],[1,1],'same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(1000, [1,1],[1,1],'same', activation=None))
    model.add(tf.keras.layers.Reshape([1000]))
    model.add(tf.keras.layers.Softmax())



    model.summary()


# vgg_16_dense()
vgg_16_txt_conv()
