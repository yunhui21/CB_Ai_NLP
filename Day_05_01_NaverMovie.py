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

# naver_ratings_test.txt, naver_ratings_test.txt

# 문제1
# 데이터를 반환하는 함수를 만드세요.

# 문제2
# 김영 박사의 코드를 사용해서 도큐먼트를 토근으로 만드세요.
# get_data 함수를 수정하세요.

# 문제3
# 도큐먼트에 포함된 단어 갯수를 그래프로 그려주세요.

# 문제4
# 도큐먼트의 길이를 동일하게 만드세요.
# 길이를 똑같이 한다. 나오면 자르고, 부족하면 채워야 한다.

# 문제5
# vocab을 만들어라(단어장)

# 문제6
# 도큐먼트를 피쳐로


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
    return documents, labels


def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^가-힣A-Za-z0-9(),!?\'\`]", " ", string)
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


def show_word_counts(documents):
    counts = [len(words) for words in documents]
    counts = sorted(counts)
    # print(counts)
    # print(len(counts))

    plt.plot(range(len(counts)), counts)
    plt.show()


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


def make_vocab(documents, vocab_size):
    tokens = [w for words in documents for w in words]
    freq = nltk.FreqDist(tokens)
    return [w for w, _ in freq.most_common(vocab_size)]


def make_feature(words, vocab):
    # feature, uniques = {}, set(words)  # unique 중복된 데이터를 제겨한다.
    uniques, feature = set(words), {}
    for v in vocab:  # vocab의 개수 2000개 이므로 feature는 2000개 생성
        feature['has_{}', format(v)] = (v in uniques)  #도큐먼트마다 다른것이므로

    return feature


def make_feature_data(documents, labels, vocab):
    # return [(make_feature(words, vocab), label) for words, label in zip(documents, labels)]
    features = [make_feature(words, vocab) for words in documents]
    return [(feat, ibl) for feat, ibl in zip(features, labels)]

x_train, y_train = get_data('data/naver_ratings_train.txt')
x_test, y_test = get_data('data/naver_ratings_test.txt')

# show_word_counts(x_train)
#
vocab = make_vocab(x_train, vocab_size=2000)

x_train = make_same_length(x_train, max_size=25)
x_test  = make_same_length(x_test, max_size=25)

train_set = make_feature_data(x_train, y_train, vocab)
test_set = make_feature_data(x_test, y_test, vocab)

clf = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(clf, train_set))



# first complete
# def get_data(file_path):
#     f = open(file_path, 'r', encoding='utf-8')
#     f.readline()
#
#     documents =[], labels =[]
#     for row in csv.reader(f, dialect='\n'):
#         #print(row)
#         documents.append(row[:])
#         labels.append(int(row[:]))
#
#     f.close()
#     return documents, labels