# Day_23_02_WordRnnChosun.py
import tensorflow as tf
import numpy as np
import re

# url = 'http://bit.ly/2mc3sov'
# file_path = tf.keras.utils.get_file('chosun.txt', url, cache_dir =',', cache_subdir='data')
# print(file_path)


def clean_str(string):

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
    long_text = f.read(100)
    long_text = clean_str(long_text)
    f.close()

    tokens = long_text.split()
    vocab = sorted(set(tokens)) + ['UNK']

    return tokens, vocab


def act_like_writer(sent, model, word2idx, idx2word, seq_length = 25):
    current = sent.split()
    for i in range(100):
        tokens = current[-seq_length:]
        token_idx = [word2idx[w] if w in word2idx else word2idx['UNK'] for w in tokens]
        # token_idx = np.array(token_idx)

        token_pad = tf.keras.preprocessing.sequence.pad_sequences(
            [token_idx], maxlen=seq_length, padding='pre', value=word2idx['UNK']
        )
        # 1번
        # output = model.predict(token_pad)
        # print(output.shape)
        # print(output)
        # output_arg = np.argmax(output, axis=1)
        # print(output_arg)

        #2 번
        output_arg = model.predict_classes(token_pad)
        # print(output_arg)

        output_arg = np.argmax[0]
        current.append(idx2word[output_arg])
        # print(output_arg)
    # print(current)


tokens, vocab = get_data()
# print(vocab)

word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = np.array(vocab)

tokens_idx = [word2idx for w in tokens]
# print(tokens[:5])
# print(tokens_idx[:5])

sent_slices = tf.data.Dataset.from_tensor_slices(tokens_idx)
# print(sent_slices)
# print(sent_slices.take(2))

for s in sent_slices.take(2):
    print(s.numpy(), s)
print()

seq_length = 25
sent_batchsize = sent_slices.batch(seq_length + 1, drop_remainder=True)

for a in sent_batchsize.take(2):
    print(a.numpy())
print()

sent_map = sent_batchsize.map(lambda chunk: (chunk[:-1], chunk[-1]))

# 문제
# sent_map 데이터를 2개만 출력하세요..

for xx, yy in sent_map.take(2):
    print(xx.numpy(), yy.numpy())
print()

sent_shuffle = sent_map.shuffle(buffer_size = 10000)

for xx, yy in sent_shuffle.take(2):
    print(xx.numpy(), yy.numpy())
print()

batch_size = 120
sent_final = sent_shuffle.batch(batch_size)

for xx, yy in sent_final.take(2):
    print(xx.numpy(), yy.numpy())
print()

model = tf.keras.Sequential([
    tf.keras.layers.Input([seq_length]),
    tf.keras.layers.Embedding(len(vocab), 100),
    tf.keras.layers.LSTM(120, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(120),
    tf.keras.layers.Dense(len(vocab), activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss = tf.keras.losses.sparse_categorical_crossentropy,
              matrics = ['acc'])

n_exemples = len(tokens) // (seq_length + 1)
steps_per_epoch = n_exemples // batch_size

model.fit(sent_final.repeat(),
          steps_per_epoch=steps_per_epoch,
          epochs= -1, verbose=2)

act_like_writer('동원에 나가 활을 쏘다', model, idx2word, word2idx, seq_length)