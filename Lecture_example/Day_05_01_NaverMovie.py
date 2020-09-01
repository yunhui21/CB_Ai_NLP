# Day_05_01_NaverMovie.py
import csv
import nltk
import gensim
import numpy as np
import re
import matplotlib.pyplot as plt

# 문제 1
# 데이터를 반환하는 함수를 만드세요 (x, y)

# 문제 2
# 김윤 박사의 코드를 사용해서 도큐먼트를 토큰으로 만드세요
# get_data 함수를 수정하세요

# 문제 3
# 도큐먼트에 포함된 단어 갯수를 그래프로 그려주세요

# 문제 4
# 도큐먼트의 길이를 동일하게 만드세요

# 문제 5
# 단어장을 만드세요 (2000개)

# 문제 6
# 도큐먼트를 피처로 변환하세요


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
    return documents, labels


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


def show_word_counts(documents):
    counts = [len(words) for words in documents]
    counts = sorted(counts)
    # print(counts)
    # print(len(counts))

    plt.plot(range(len(counts)), counts)
    plt.show()


def make_same_length(documents, max_size):
    docs = []
    for words in documents:
        if len(words) < max_size:
            docs.append(words + [''] * (max_size - len(words)))
        else:
            docs.append(words[:max_size])

        assert(len(docs[-1]) == max_size)

    return docs


def make_vacab(documents, vocab_size):
    tokens = [w for words in documents for w in words]
    freq = nltk.FreqDist(tokens)
    return [w for w, _ in freq.most_common(vocab_size)]


def make_feature(words, vocab):
    uniques, feature = set(words), {}
    for v in vocab:
        feature['has_{}'.format(v)] = (v in uniques)

    return feature


def make_feature_data(documents, labels, vocab):
    # return [(make_feature(words, vocab), label) for words, label in zip(documents, labels)]

    features = [make_feature(words, vocab) for words in documents]
    return [(feat, lbl) for feat, lbl in zip(features, labels)]


x_train, y_train = get_data('data/naver_ratings_train.txt')
x_test, y_test = get_data('data/naver_ratings_test.txt')

# 대랙 40개의 단어라면 대부분을 포함하는 것을 확인
# show_word_counts(x_train)

vocab = make_vacab(x_train, vocab_size=2000)

x_train = make_same_length(x_train, max_size=25)
x_test = make_same_length(x_test, max_size=25)

train_set = make_feature_data(x_train, y_train, vocab)
test_set = make_feature_data(x_test, y_test, vocab)

clf = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(clf, test_set))

