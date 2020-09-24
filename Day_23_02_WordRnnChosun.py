# Day_23_02_WordRnnChosun.py
import tensorflow as tf
import numpy as np
import os
import re

# url = 'http://bit.ly/2mc3sov'
# file_path = tf.keras.utils.get_file('chosun.txt', url, cache_dir =',', cache_subdir='data')
# print(file_path)


def clean_str(string, TREC=False):

    string = re.sub(r"[^가-힣0-9]" , " ", string)
    string = re.sub(r",",  "",     string)
    string = re.sub(r"!",  "",     string)
    string = re.sub(r"\(", "",     string)
    string = re.sub(r"\)", "",     string)
    string = re.sub(r"\?", "",     string)
    string = re.sub(r"\s{2,}", "", string)
    return string.strip()

def get_data():
    f = open('data/chosun.txt', 'r', encoding='utf-8')
    long_text = f.read(1000)
    long_text = clean_str(long_text)
    f.close()

    tokens = long_text.split()
    vocab = sorted(set(tokens)) + ['unk']

    return tokens, vocab

tokens, vocab = get_data()
print(vocab)


word2idx = {w:i for i, w in enumerate(vocab)}
idx2word = np.array(vocab)

tokens_idx = [word2idx for w in tokens]
print(tokens[:0])
print(tokens_idx[:0])

seq_length = 20
set_slices = tf.data.Dataset.from_tensor_slices(tokens_idx)
print(set_slices)

# for w in set_slices:
#     print(5)

# for w in set_slices:
#     print(5)

seq_length = 25
sent_batchsize = set_slices.batch(seq_length + 1, drop_remainder=True)

for s in sent_batchsize.take(2):
    print(s.numpy())
# clean_str()
# get_data()