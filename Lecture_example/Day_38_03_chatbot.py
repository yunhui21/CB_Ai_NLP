# Day_38_03_chatbot.py
import tensorflow as tf
import numpy as np
import nltk
import csv

_PAD_, _SOS_, _EOS_, _UNK_ = 0, 1, 2, 3


def load_vocab(data_folder):
    path = '{}/vocab.txt'.format(data_folder)
    f = open(path, 'r', encoding='utf-8')
    vocab = nltk.word_tokenize(f.read())
    f.close()

    return vocab        # 리스트의 index 함수 사용을 위해 numpy 변환 안함


def load_vectors(data_folder):
    path = '{}/vectors.txt'.format(data_folder)
    f = open(path, 'r', encoding='utf-8')
    vectors = [[int(i) for i in row] for row in csv.reader(f)]
    f.close()

    # print(vectors[5])     # [125, 118, 37, 49]
    return vectors          # 길이가 들쭉날쭉하기 때문에 numpy로 변환 불가능


def add_pad(seq, max_len):
    seq_len = len(seq)
    if seq_len >= max_len:
        # assert(seq_len == max_len)
        return seq[:max_len]

    return seq + [_PAD_] * (max_len - seq_len)


def load_dataset(data_folder):
    vocab = load_vocab(data_folder)
    vectors = load_vectors(data_folder)

    # 문제
    # 질문과 대답으로부터 가장 긴 문장의 길이를 구하세요
    max_len_enc = max([len(s) for s in vectors[::2]])
    max_len_dec = max([len(s) for s in vectors[1::2]])
    print(max_len_enc, max_len_dec)     # 9 9

    # 앞에서 만들었던 make_batch 함수와 거의 비슷하다
    onehots = np.eye(len(vocab), dtype=np.float32)

    enc_inputs, dec_inputs, dec_target = [], [], []
    for i in range(0, len(vectors), 2):
        question, answer = vectors[i], vectors[i+1]

        # 문제
        # dec_in과 target 코드를 만드세요
        enc_in = add_pad(question, max_len_enc)
        dec_in = add_pad([_SOS_]+answer, max_len_dec)
        target = add_pad(answer+[_EOS_], max_len_dec)

        enc_inputs.append(onehots[enc_in])
        dec_inputs.append(onehots[dec_in])
        dec_target.append(target)           # sparse

    return np.float32(enc_inputs), np.float32(dec_inputs), np.float32(dec_target), vocab


def train_and_save(enc_inputs, dec_inputs, dec_target, n_classes, model_folder):
    n_hiddens = 128

    enc_inputs_layer = tf.keras.layers.Input([None, n_classes])
    _, enc_states = tf.keras.layers.SimpleRNN(
        n_hiddens, return_state=True)(enc_inputs_layer)

    dec_inputs_layer = tf.keras.layers.Input([None, n_classes])
    dec_output = tf.keras.layers.SimpleRNN(
        n_hiddens, return_sequences=True)(dec_inputs_layer, initial_state=enc_states)

    outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(dec_output)

    model = tf.keras.Model([enc_inputs_layer, dec_inputs_layer], outputs)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.sparse_categorical_crossentropy)

    model.fit([enc_inputs, dec_inputs], dec_target, epochs=500)

    model.save('{}/chat.h5'.format(model_folder))


enc_inputs, dec_inputs, dec_target, vocab = load_dataset('chat_data')

train_and_save(enc_inputs, dec_inputs, dec_target, len(vocab), 'chat_model')



