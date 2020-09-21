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

'''
* * * * * * *
* * * * * * *
* * * * * * *
* * * * * * *
* * * * * * *

7x7 : 



5x5 : 3x3(9)
3x3 : 5x5(25)      - 수용능력을 유지할수있다.
      3x3 = 3x3(9) - 연산량이 줄어 든다.

1x5 : 7x3(21)
      5x1: 3x3(9)

** Residual learnin gbuilding block
*+ h(x)를 얻도록 학습하는 것이 아니라 H(x)-x를 얻도록 학습 방향 변경
** F(x) +  x에서 f(x)는 0이 되려고 한다.
** concat 기능
** https://curaai00.tistory.com/1 resNet:residual learning for image recognition (paper)
'''

