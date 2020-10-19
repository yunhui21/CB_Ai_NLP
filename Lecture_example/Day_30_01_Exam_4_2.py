# Day_30_01_Exam_4_2.py
import json
import tensorflow as tf
import numpy as np
from sklearn import model_selection

# 문제 4
# sarcasm.json 파일을 읽어서 2만개로 학습하고 나머지에 대해 정확도를 예측하세요
# 목표 : 80%
# (이전 문제에 다양한 제약이 가해집니다. 미리 제공한 변수를 모두 사용하세요)


def get_xy():
    f = open('data/sarcasm.json', 'r', encoding='utf-8')
    data = json.load(f)
    f.close()

    x, y = [], []
    for item in data:
        x.append(item['headline'])
        y.append(item['is_sarcastic'])

    return x, np.int32(y)


x, y = get_xy()

vocab_size = 1000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_size = 20000

# --------------------------------------------- #

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(x)

x = tokenizer.texts_to_sequences(x)

x = tf.keras.preprocessing.sequence.pad_sequences(x,
                                                  maxlen=max_length,
                                                  padding=padding_type,
                                                  truncating=trunc_type)

data = model_selection.train_test_split(x, y, train_size=training_size)
x_train, x_valid, y_train, y_valid = data

model = tf.keras.Sequential()
model.add(tf.keras.layers.Input([max_length]))
model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim))
model.add(tf.keras.layers.Conv1D(32, 5, activation='relu'))
model.add(tf.keras.layers.MaxPool1D())
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)))
model.add(tf.keras.layers.Dense(1))

model.summary()

model.compile(optimizer=tf.keras.optimizers.RMSprop(0.0003),
              loss=tf.keras.losses.binary_crossentropy,
              metrics=['acc'])
model.fit(x_train, y_train, epochs=5, verbose=2, batch_size=128,
          validation_data=(x_valid, y_valid))



