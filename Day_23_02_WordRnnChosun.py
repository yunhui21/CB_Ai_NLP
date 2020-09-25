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
    long_text = f.read(100000)
    long_text = clean_str(long_text)
    f.close()

    tokens = long_text.split()
    vocab = sorted(set(tokens)) + ['UNK']

    return tokens, vocab


def act_like_writer(sent, model, word2idx, idx2word, seq_length=25):
    current = sent.split()

    for i in range(100):
        tokens = current[-seq_length:]
        token_idx = [word2idx[w] if w in word2idx else word2idx['UNK'] for w in tokens]

        token_pad = tf.keras.preprocessing.sequence.pad_sequences(
            [token_idx], maxlen=seq_length, padding='pre', value=word2idx['UNK']
        )

        # 1번
        # output = model.predict(token_pad)
        # # print(output.shape)     # (1, 8649)
        # # print(output)           # [[0. 0. 0. ... 0. 0. 0.]]
        #
        # output_arg = np.argmax(output, axis=1)
        # # print(output_arg)       # [3326]

        # 2번
        output_arg = model.predict_classes(token_pad)
        # print(output_arg)         # [3326]

        output_arg = output_arg[0]
        current.append(idx2word[output_arg])

    print(current)

def model_chosun():
    tokens, vocab = get_data()
    print(vocab)

    word2idx = {w:i for i, w in enumerate(vocab)}
    idx2word = np.array(vocab)

    tokens_idx = [word2idx[w] for w in tokens]
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
    for xx, yy in sent_map.take(2):
        print(xx.numpy(), yy.numpy())
    print()

    sent_shuffle = sent_map.shuffle(buffer_size=10000)

    for xx, yy in sent_shuffle.take(2):
        print(xx.numpy(), yy.numpy())
    print()

    batch_size = 128
    sent_final = sent_shuffle.batch(batch_size)

    for xx, yy in sent_final.take(2):
        print(xx.shape, yy.shape)       # (128, 25) (128,)
    print()

    model = tf.keras.Sequential([
        tf.keras.layers.Input([seq_length]),
        tf.keras.layers.Embedding(len(vocab), 100),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dense(len(vocab), activation='softmax'),
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])

    n_examples = len(tokens) // (seq_length + 1)
    steps_per_epoch = n_examples // batch_size
    print('steps_per_epoch :', steps_per_epoch)

    model.fit(sent_final.repeat(),
              steps_per_epoch=steps_per_epoch,
              epochs=20, verbose=2)

    act_like_writer('동헌에 나가 활을 쏘다', model, word2idx, idx2word, seq_length)

def load_chosun():

    tokens, vacab = get_data()

    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = np.array(vocab)

    model = tf.keras.models.load_model('data/chousun.100.h5')
    act_like_writer('동현에 나가 활을 쏘다.', model, word2idx, idx2word, seq_length=25)

# model_chosun()
load_chosun()