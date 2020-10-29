# Day_38_03_chatbot.py
import tensorflow as tf
import numpy as np
import nltk
import csv
import sys

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
    max_len_dec = max([len(s) for s in vectors[1::2]]) + 1 # +1 은 tag
    print(max_len_enc, max_len_dec)     # 9 9

    # 앞에서 만들었던 make_batch 함수와 거의 비슷하다
    onehots = np.eye(len(vocab), dtype=np.float32)

    enc_inputs, dec_inputs, dec_target = [], [], []
    for i in range(0, len(vectors), 2):
        question, answer = vectors[i], vectors[i+1]
        print() # 9
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

def load_and_predict(enc_inputs, dec_inputs, dec_target, vocab, model_folder):
    model = tf.keras.models.load_model('{}/chat.h5'.format(model_folder))
    preds = model.predict([enc_inputs, dec_inputs], verbose=0)
    preds_arg = np.argmax(preds, axis=2) # ,제일 마지막 결과값을 갖고 오기 위해 2
    print(preds_arg)

    # 문제
    # preds_arg를 문자열로 변환해서 출력하세요.
    # [ 45  50 148 116   2   0   0   0   0] 2: eos 앞까지만 값을 찾아서 변환한다.
    for question, answer, result in zip(enc_inputs,dec_target, preds_arg):
        # print(question) # onehot이라서 단순인코딩으로 변환
        # print(np.argmax(question, axis=1))
        question = np.argmax(question, axis=1)
        convert_to_sentence(np.int32(question), vocab, is_question=True)
        convert_to_sentence(np.int32(answer), vocab, is_question=False)
        convert_to_sentence(result, vocab, is_question=False)
        print()

def convert_to_sentence(sent, vocab, is_question):

    # sent = preds_arg[1]
    sent = list(sent)
    # print(sent)

    # 문제
    # numpy
    if is_question:
        pos = sent.index(_EOS_) if _EOS_ in sent else len(sent)
    else:
        pos = sent.index(_EOS_)

    pos = len(sent) if is_question else sent.index(_EOS_)
    # print(pos)


    # print([i for i in sent[:pos]])
    # print([vocab[i] for i in sent[:pos]])
    result = ' '.join(vocab[i] for i in sent[:pos])
    result = result.replace(vocab[_PAD_], '')

    print('왕자:' if is_question else '여우:', result )


def talk_to_bot(data_folder, model_folder):
    vocab = load_vocab(data_folder)
    vectors = load_vectors(data_folder)

    max_len_enc = max([len(s) for s in vectors[::2]])
    max_len_dec = max([len(s) for s in vectors[1::2]]) + 1  # +1 은 tag
    # print(max_len_enc, max_len_dec)  # 9 9

    onehots = np.eye(len(vocab), dtype=np.float32)

    model = tf.keras.models.load_model('{}/chat.h5'.format(model_folder))

    while type:
        sys.stdout.write('왕자:')
        sys.stdout.flush()
        line = sys.stdin.readline()
        if '끝' in line:
            break

        # line = '이리 와서 나하고 놀자'

        # 문제
        # 입력받은 문자열을 사용해서 결과를 예측하세요.
        # tokens = line.split()
        tokens = nltk.regexp_tokenize(line, '\w+')
        # print(tokens)

        question = [vocab.index(t) if t in vocab else _UNK_ for t in tokens]

        enc_in = add_pad(question, max_len_enc)
        dec_in = add_pad([_SOS_], max_len_dec)

        enc_inputs = np.int32([onehots[enc_in]])
        dec_inputs = np.int32([onehots[dec_in]])

        preds = model.predict([enc_inputs, dec_inputs])
        preds_arg = np.argmax(preds, axis=2)

        equale = np.where(preds_arg[0] == _EOS_)
        # print(equale)
        # print(equale[0])
        convert_to_sentence(preds_arg[0], vocab, is_question=False)

        # break
enc_inputs, dec_inputs, dec_target, vocab = load_dataset('chat_data')

# train_and_save(enc_inputs, dec_inputs, dec_target, len(vocab), 'chat_model')
# load_and_predict(enc_inputs, dec_inputs, dec_target, vocab, 'chat_model')
talk_to_bot('chat_data', 'chat_model')

