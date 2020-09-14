# Day_15_02_NavermMovie.py

# Day_05_01_NaverMovie.py
# 네이버 영화    검색
# CNN sentence 검색 https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
import csv
import nltk
import gensim
import numpy as np
import collections
import re
import matplotlib.pyplot as plt
import tensorflow as tf

# naver_ratings_test.txt, naver_ratings_test.txt

def get_data(file_path):
    f = open(file_path, 'r', encoding='utf-8')
    f.readline()

    documents, labels =[], []
    for row in csv.reader(f, delimiter='\t'):
        # print(row)
        # print(row[1], clean_str((row[1])))

        documents.append(clean_str(row[1]).split())
        labels.append(int(row[2]))
        # print(documents)

    f.close()
    small = int(len(documents)*0.3)
    return documents[:small], labels[:small]


def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^가-힣A-Za-z0-9(),!?\'\'] " , " ", string)
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


def make_same_length(documents, max_size):
    #원본 수정, 반환값 사용. 2가지중 선택
    docs = []
    for words in documents:
        if len(words) < max_size:
            docs.append(words + ['']*(max_size - len(words)))
        else:
            docs.append(words[:max_size])

        assert(len(docs[-1]) == max_size)
    return docs


x_train, y_train = get_data('data/naver_ratings_train.txt')
x_test, y_test = get_data('data/naver_ratings_test.txt')

y_train = np.reshape(y_train, [-1,1])
y_test = np.reshape(y_test, [-1,1])

# x_train = make_same_length(x_train, max_size=25)
# x_test  = make_same_length(x_test, max_size=25)

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


# 여러분의 리뷰가 긍정인지 부정인지 단계별로 알려주세요.
#

new_review = '지금껏 본것중에서 가장 감동적인 영화...'
tokens = clean_str(new_review).split()
print(tokens)

x_review = tokenizer.texts_to_sequences([tokens])       #1차원 -> 2차원
x_review = x_review[0]                                  #2차원 -> 1차원
print(x_review)

for i in range(len(x_review)):
    words = [x_review[:i+1]]
    words = tf.keras.preprocessing.sequence.pad_sequences(words, padding='post', maxlen=25)

    print([tokenizer.index_word[k] for k in x_review[:i+1]])
    # print(x_review[:i+1])
    # print(tokenizer.index_word[x_review[:i+1]])# index_word numpy가 되어야 한다.
    # print(tokens[:i+1])
    print(model.predict(words, verbose=0))
