# Day_15_02_RnnNaverMovie.py
import csv
import numpy as np
import re
import tensorflow as tf


def get_data(file_path):
    f = open(file_path, 'r', encoding='utf-8')
    f.readline()

    documents, labels = [], []
    for row in csv.reader(f, delimiter='\t'):
        # print(row)
        # print(row[1], clean_str(row[1]))
        documents.append(clean_str(row[1]).split())
        labels.append(int(row[2]))

    f.close()
    # return documents, labels

    small = int(len(documents) * 0.3)
    return documents[:small], labels[:small]


def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^가-힣A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


def make_same_length(documents, max_size):
    docs = []
    for words in documents:
        if len(words) < max_size:
            docs.append(words + [''] * (max_size - len(words)))
        else:
            docs.append(words[:max_size])

        assert(len(docs[-1]) == max_size)

    return docs


x_train, y_train = get_data('data/naver_ratings_train.txt')
x_test, y_test = get_data('data/naver_ratings_test.txt')

y_train = np.reshape(y_train, [-1, 1])
y_test = np.reshape(y_test, [-1, 1])

# x_train = make_same_length(x_train, max_size=25)
# x_test = make_same_length(x_test, max_size=25)

vocab_size = 2000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(x_train)

# print(tokenizer.index_word)

x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)
print(x_train[0])

x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, padding='post', maxlen=25)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, padding='post', maxlen=25)
print(x_train[0])

print(x_train.shape, y_train.shape)     # (150000, 25) (150000, 1)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 100),
    tf.keras.layers.LSTM(50),
    tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid),
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss=tf.keras.losses.binary_crossentropy,
              metrics=['acc'])
model.fit(x_train, y_train, epochs=5, verbose=2, batch_size=128)
print('acc :', model.evaluate(x_test, y_test, verbose=2))

# 문제
# 여러분의 리뷰가 긍정인지 부정인지 단계별로 알려주세요
# 단어 갯수를 늘려가면서 결과를 보여주세요
new_review = '친구가 재밌다 그랬는데.. 뻥쟁이! 핵노잼'

tokens = clean_str(new_review).split()
print(tokens)
x_review = tokenizer.texts_to_sequences([tokens])   # 1차원 -> 2차원
x_review = x_review[0]                              # 2차원 -> 1차원
print(x_review)

for i in range(len(x_review)):
    words = [x_review[:i+1]]        # 1차원 -> 2차원
    words = tf.keras.preprocessing.sequence.pad_sequences(words, padding='post', maxlen=25)

    print([tokenizer.index_word[k] for k in x_review[:i+1]])
    # print(tokenizer.index_word[x_review[:i+1]])   # error
    # print(tokens[:i+1])                           # wrong
    print(model.predict(words, verbose=0))

# (10000, 784) -> (784, 256) -> (256, 128) -> (128, 10)


