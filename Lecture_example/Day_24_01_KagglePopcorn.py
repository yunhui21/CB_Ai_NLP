# Day_24_01_KagglePopcorn.py
import tensorflow as tf
import numpy as np
import re
import pandas as pd
from sklearn import model_selection, feature_extraction, linear_model
import matplotlib.pyplot as plt

# 문제 1
# 팝콘 데이터를 8대2로 나눠서 예측하세요

# 문제 2
# 예측한 결과를 서브미션 파일로 만들어서 캐글에 업로드하세요
# (my_popcorn_tfidf.csv)

# 문제 3
# tf-idf에서 만든 코드를 활용해서 베이스라인 모델 서브미션도 업로드하세요


def tokenizing_and_padding(x, vocab_size, seq_length):
    # vocab_size = 2000
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(x)

    # print(tokenizer.num_words)          # None일 때 아래처럼 사용
    # print(len(tokenizer.index_word))    # 88582 (결과에 1을 더해서 사용함, 실제 단어장에는 reverved 단어 포함됨)

    x = tokenizer.texts_to_sequences(x)

    # seq_length = 200
    x = tf.keras.preprocessing.sequence.pad_sequences(x, padding='post', maxlen=seq_length)

    return x, tokenizer


def model_baseline(x, y, ids, x_test):
    # 문제
    # 문장에 포함된 단어 갯수를 시각화하세요
    # lengths = sorted([len(tokens) for tokens in x], reverse=True)
    # plt.plot(range(len(lengths)), lengths)
    # plt.show()

    vocab_size = 2000
    x, tokenizer = tokenizing_and_padding(x, vocab_size=vocab_size, seq_length=200)

    data = model_selection.train_test_split(x, y, train_size=0.8, test_size=0.2, shuffle=False)
    x_train, x_valid, y_train, y_valid = data

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 100),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid),
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['acc'])
    model.fit(x_train, y_train, epochs=5, verbose=2, batch_size=128,
              validation_data=[x_valid, y_valid])
    # print('acc :', model.evaluate(x_valid, y_valid, verbose=2))

    # ------------------------------------- #

    x_test = tokenizer.texts_to_sequences(x_test)
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, padding='post', maxlen=200)

    preds = model.predict(x_test)
    preds_arg = preds.reshape(-1)
    preds_bool = np.int32(preds_arg > 0.5)

    with open('word2vec-popcorn/my_popcorn_baseline.csv', 'w', encoding='utf-8') as f:
        f.write('"id","sentiment"\n')
        for i, p in zip(ids, preds_bool):
            f.write('"{}",{}\n'.format(i, p))


def model_tfidf(x, y, ids, x_test):
    x, tokenizer = tokenizing_and_padding(x, vocab_size=2000, seq_length=200)   # 88582

    # 숫자를 문자열 토큰으로 변환 (숫자에 lower 함수를 호출하는 과정에서 에러)
    x = tokenizer.sequences_to_texts(x)
    tfidf = feature_extraction.text.TfidfVectorizer(
        min_df=0.0, analyzer='word', sublinear_tf=True, ngram_range=(1, 3), max_features=5000)
    x = tfidf.fit_transform(x)

    data = model_selection.train_test_split(x, y, train_size=0.8, test_size=0.2, shuffle=False)
    x_train, x_valid, y_train, y_valid = data

    # ----------------------------- #

    lr = linear_model.LogisticRegression(class_weight='balanced')
    lr.fit(x_train, y_train)
    print('acc :', lr.score(x_valid, y_valid))

    # ----------------------------- #

    # preds = lr.predict(x_test)
    # print(preds.shape)
    # print(preds[:5])
    # print(y_test[:5])
    #
    # print('acc :', np.mean(preds == y_test.reshape(-1)))

    x_test = tokenizer.texts_to_sequences(x_test)
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, padding='post', maxlen=200)

    x_test = tokenizer.sequences_to_texts(x_test)
    x_test = tfidf.transform(x_test)

    preds = lr.predict(x_test)

    with open('word2vec-popcorn/my_popcorn_tfidf.csv', 'w', encoding='utf-8') as f:
        f.write('"id","sentiment"\n')
        for i, p in zip(ids, preds):
            f.write('"{}",{}\n'.format(i, p))


popcorn = pd.read_csv('word2vec-popcorn/labeledTrainData.tsv',
                      delimiter='\t',
                      index_col=0)
# print(popcorn.head())

x = popcorn.review
# y = popcorn.sentiment.reshape(-1, 1)          # error
# y = np.reshape(popcorn.sentiment, [-1, 1])    # error
y = popcorn.sentiment.values.reshape(-1, 1)     # good

n_samples = None
if n_samples:
    x = x[:n_samples]
    y = y[:n_samples]

test_set = pd.read_csv('word2vec-popcorn/testData.tsv',
                       delimiter='\t',
                       index_col=0)
ids = test_set.index.values
x_test = test_set.review

model_baseline(x, y, ids, x_test)
# model_tfidf(x, y, ids, x_test)


# (baseline)  0.83
# (tf-idf  )  0.8738


