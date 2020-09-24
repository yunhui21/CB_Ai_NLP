# Day_23_02_WordRnnChosun.py
import tensorflow as tf
import numpy as np
import re
tf.compat.v1.enable_eager_execution()

# url = 'http://bit.ly/2Mc3SOV'
# file_path = tf.keras.utils.get_file('chosun.txt', url, cache_dir='.', cache_subdir='data')
# print(file_path)


def clean_str(string):
    string = re.sub(r"[^가-힣0-9]", " ", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"!", "", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


def get_data():
    f = open('data/chosun.txt', 'r', encoding='utf-8')
    long_text = f.read(1000)
    long_text = clean_str(long_text)
    f.close()

    tokens = long_text.split()
    vocab = sorted(set(tokens)) + ['UNK']

    return tokens, vocab


tokens, vocab = get_data()
print(vocab)

word2idx = {w:i for i, w in enumerate(vocab)}
idx2word = np.array(vocab)

tokens_idx = [word2idx for w in tokens]
print(tokens[:5])           # ['태조', '이성계', '선대의', '가계', '목조']
print(tokens_idx[:5])       # [135, 101, 66, 6, 49]

sent_slices = tf.data.Dataset.from_tensor_slices(tokens_idx)
print(sent_slices)          # <DatasetV1Adapter shapes: (), types: tf.int32>
print(sent_slices.take(2))  # <DatasetV1Adapter shapes: (), types: tf.int32>

for s in sent_slices.take(2):
    print(s.numpy(), s)         # 135, 101, ...
print()

seq_length = 25                 # x           y
sent_batches = sent_slices.batch(seq_length + 1, drop_remainder=True)

for s in sent_batches.take(2):
    print(s.numpy())
print()

sent_map = sent_batches.map(lambda chunk: (chunk[:-1], chunk[-1]))

# 문제
# sent_map 데이터를 2개만 출력하세요

