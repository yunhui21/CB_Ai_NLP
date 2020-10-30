# Day_40_03_attentionimdb.py

import tensorflow as tf
import numpy as np
from Day_40_02_attentionlayer import Attention, BahdanauAttention

# 문제
# 에포크마다 발생하는 validation의 정확도를 저장해서
# 학습이 끝난이후에    평균의 최대 정학도를 출력하세요.
class validsave(tf.keras.callbacks.Callback):
    def __init__(self):
        self.val_acc = []

    def on_epoch_end(self, epoch, logs=None):
        self.val_acc.append(logs['val_acc'])

def show_attention(option):
    num_words, max_len, embed_size = 5000, 500, 32

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words = num_words)
    print(x_train.shape, x_test.shape) # (25000,) (25000,)
    print(y_train.shape, y_test.shape) # (25000,) (25000,)

    # print(x_train[:3]) # [list([1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, 66, 3941, 4, 173, 36,...])]

    # print(y_train[:10]) # [1 0 0 1 0 0 1 0 1 0]
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
    x_test  = tf.keras.preprocessing.sequence.pad_sequences(x_test,  maxlen=max_len)
    print(x_train.shape, x_test.shape) # (25000, 500) (25000, 500)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input([max_len]))
    model.add(tf.keras.layers.Embedding(num_words, embed_size)) # 전이학습이 효과적
    model.add(tf.keras.layers.LSTM(100)) # 레이어 개수를 늘이거나, bidirectional을 적용 효과적
    model.add(tf.keras.layers.Dense(350, activation='relu'))

    if option == 0:
        model.add(tf.keras.layers.LSTM(100))
        model.add(tf.keras.layers.Dense(350, activation='relu'))

    elif option == 1:
        model.add(tf.keras.layers.LSTM(100, return_sequences=True))
        model.add(Attention())
        # model.add(tf.keras.layers.LSTM(100, return_sequences=True)) # decoder를 정의할때 형식

    else:
        model.add(tf.keras.layers.LSTM(100, return_sequences=True))
        model.add()

    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss= tf.keras.losses.binary_crossentropy,
                  metrics=['acc'])
    valid_save = validsave()
    model.fit(x_train, y_train, epochs=3, batch_size= 1024,
              validation_data=(x_test, y_test),
              callbacks= [valid_save]) # []로 묶으면 bug val = 0

    print('acc_avg:', np.mean(valid_save.val_acc[-3:]))
    print('acc_avg:', np.max(valid_save.val_acc[-3:]))

# show_attention(option =0)
# show_attention(option = 1)
show_attention(option = 2)