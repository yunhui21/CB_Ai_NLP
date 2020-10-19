# Day_30_01_Exam_4_1.py
import urllib.request
import json
import tensorflow as tf
import numpy as np
from sklearn import model_selection

# 문제 4
# sarcasm.json 파일을 읽어서 2만개로 학습하고 나머지에 대해 정확도를 예측하세요
# 목표 : 80%

# url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
# urllib.request.urlretrieve(url, 'data/sarcasm.json')


def get_xy():
    f = open('data/sarcasm.json', 'r', encoding='utf-8')
    data = json.load(f)
    f.close()

    print(type(data))
    print(len(data))        # 26709

    x, y = [], []
    for item in data:
        # print(type(item), item.keys())
        # <class 'dict'> dict_keys(['article_link', 'headline', 'is_sarcastic'])

        x.append(item['headline'])
        y.append(item['is_sarcastic'])
        # print(type(y[-1]))

    return x, np.int32(y)


x, y = get_xy()

vocab_size = 2000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(x)

x = tokenizer.texts_to_sequences(x)

seq_length = 200
x = tf.keras.preprocessing.sequence.pad_sequences(x, padding='post', maxlen=seq_length)

data = model_selection.train_test_split(x, y, train_size=20000)
x_train, x_valid, y_train, y_valid = data

model = tf.keras.Sequential()
model.add(tf.keras.layers.Input([seq_length]))
model.add(tf.keras.layers.Embedding(vocab_size, 100))
model.add(tf.keras.layers.Conv1D(32, 5, activation='relu'))
model.add(tf.keras.layers.MaxPool1D())
# model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)))
# model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32, dropout=0.2, recurrent_dropout=0.2)))
model.add(tf.keras.layers.Dense(1))

model.summary()

model.compile(optimizer=tf.keras.optimizers.RMSprop(0.0003),
              loss=tf.keras.losses.binary_crossentropy,
              metrics=['acc'])
model.fit(x_train, y_train, epochs=5, verbose=2, batch_size=128,
          validation_data=(x_valid, y_valid))



