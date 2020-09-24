# Day_24_01_KagglePopcorn.py
import tensorflow as tf
import numpy as np
import pandas as pd
import re
from sklearn import feature_extraction, model_selection, linear_model
import matplotlib.pyplot as plt

# 문제
# popcorn 데이터를 예측하세요.

# 1) x, y데이터 만들기
def model_baseline(popcorn, n_samples= -1):

    x = popcorn.review
    # y = popcorn.sentiment.reshape(-1,1) # error
    # y = np.reshape(popcorn.sentiment,[-1,1]) #error
    y = popcorn.sentiment.values.reshape(-1,1) #error

    # sampling
    if n_samples > 0:
        x = x[:n_samples]
        y = y[:n_samples]

    vocab_size = 2000
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(x)

    x = tokenizer.texts_to_sequences(x)
    # print(x[0])
    # 문제
    # 문장에 포함된 단어 갯수를 시각화하세요.
    # lengths = sorted([len(tokens) for tokens in x], reverse=True)
    # plt.plot(range(len(lengths)), lengths)
    # plt.show()

    # exit(-1)

    seq_length = 200
    x = tf.keras.preprocessing.sequence.pad_sequences(x, padding='post', maxlen=seq_length)
    # print(x.shape, y.shape) # (1000, 25) (1000, 1)
    data = model_selection.train_test_split(x,y, train_size=0.8, test_size=0.2, shuffle=False)
    x_train, y_train, x_test, y_test = data

    model = tf.keras.Sequential([
                tf.keras.layers.Embedding(vocab_size, 200),
                tf.keras.layers.LSTM(50),
                tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)
            ])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                  loss=tf.keras.losses.binary_crossentropy)
    model.fit(x_train, y_train, epochs=5,  batch_size=128, verbose=2)
    print('acc:', model.evaluate(x_test, y_test))


def model_tfidf(popcorn, n_samples= -1):

    x = popcorn.review
    # y = popcorn.sentiment.reshape(-1,1) # error
    # y = np.reshape(popcorn.sentiment,[-1,1]) #error
    y = popcorn.sentiment.values.reshape(-1,1) #error

    # sampling
    if n_samples > 0:
        x = x[:n_samples]
        y = y[:n_samples]

    vocab_size = 2000
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(x)

    x = tokenizer.texts_to_sequences(x)

    seq_length = 200
    x = tf.keras.preprocessing.sequence.pad_sequences(x, padding='post', maxlen=seq_length)

    # 숫자를 문자열 토큰으로 변환(숫자에 lower 함수를 호출하는 과정에서 에러
    x = tokenizer.sequences_to_texts(x)
    tfidf = feature_extraction.text.TfidfVectorizer(
        min_df=0.0, analyzer= 'word', sublinear_tf=True, ngram_range=(1,3), max_features=5000 )
    x = tfidf.fit_transform(x)

    data = model_selection.train_test_split(x,y, train_size=0.8, test_size=0.2, shuffle=False)
    x_train, y_train, x_test, y_test = data

    lr = linear_model.logistic_regression_path(class_weight='balanced')
    lr.fit(x_train, y_train)
    print('acc:', lr.score(x_test, y_test))


popcorn = pd.read_csv('data/word2vec-popcorn/labeledTrainData.tsv', '\t', index_col=0)
print(popcorn.head())

# model_baseline(popcorn, n_samples=1000)
model_tfidf(popcorn, n_samples=1000)