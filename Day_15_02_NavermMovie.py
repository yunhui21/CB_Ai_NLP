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
# x_test, y_test = get_data('data/naver_ratings_test.txt')

x_train = make_same_length(x_train, max_size=25)
# x_test  = make_same_length(x_test, max_size=25)


