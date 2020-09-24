# Day_24_01_KagglePopcorn.py
import tensorflow as tf
import numpy as np
import pandas as pd
import re

# 문제
# popcorn 데이터를 예측하세요.

# 1) x, y데이터 만들기

def get_data(file_path):
    f = open(file_path)
    f.readline()

    review, sentiments =[], []
    for row in csv.reader(f, delimiter='\t'):
        # print(row)
        # print(row[1], clean_str((row[1])))
        review.append(clean_str(row[1]).split())
        sentiments.append(int(row[2]))
        # print(documents)

    f.close()
    small = int(len(review)*0.3)
    return review[:small], sentiments[:small]


def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\'] " , " ", string)
    string = re.sub(r"\'s", " \'s",   string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d",   string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ",      string)
    string = re.sub(r"!", " ! ",      string)
    string = re.sub(r"\(", " \( ",    string)
    string = re.sub(r"\)", " \) ",    string)
    string = re.sub(r"\?", " \? ",    string)
    string = re.sub(r"\s{2,}", " ",   string)
    return string.strip() if TREC else string.strip().lower()

def make_same_length(review, max_size):
    #원본 수정, 반환값 사용. 2가지중 선택
    docs = []
    for words in review:
        if len(words) < max_size:
            docs.append(words + ['']*(max_size - len(words)))
        else:
            docs.append(words[:max_size])

        assert(len(docs[-1]) == max_size)
    return docs

x_train, y_train = get_data('data/word2vec-popcorn/labeledTrainData.tsv')
x_test, y_test = get_data('data/word2vec-popcorn/labeledTrainData.tsv')

y_train = np.reshape(y_train, [-1,1])
y_test = np.reshape(y_test, [-1,1])

vocab_size = 2000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(x_train)

print(tokenizer.index_word)

x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)
print(x_train[0])

x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, padding='post', maxlen=25)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, padding='post', maxlen=25)

print(x_train[0])

print(x_train.shape, y_train.shape)     # (150000, 25) (150000, 1) lstm 3차원
# tensor
# ensor = (5,6)
model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, 100), #3차원
            tf.keras.layers.LSTM(50), #rnn을 사용..사용한 데이터는 시계열 형태이지만 그렇다고 할수는 없다. 2차원으로 변환하므로 그냥 빼면 안됨..2차원으로 3차원으로 변환해주는걸 넣어야 한다.
            tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)
        ])

model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss=tf.keras.losses.binary_crossentropy)
model.fit(x_train, y_train, epochs=5, verbose=2, batch_size=128)
print('acc:', model.evaluate(x_test, y_test))

# model = tf.keras.Sequential()
#
# model


