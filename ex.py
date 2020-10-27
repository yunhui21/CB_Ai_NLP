# Day_37_01_seq2seq.py
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def make_vocab_and_index(data):
    # 문제
    # 영어 알파벳과 한글 글자로 분리하세요
    eng = sorted({c for w, _ in data for c in w})
    kor = sorted({c for _, w in data for c in w})

    print(eng)  # ['a', 'b', 'd', 'e', 'f', 'h', 'i', 'l', 'm', 'n', 'o', 'p', 'r', 'u', 'w']
    print(kor)  # ['구', '나', '람', '랑', '무', '바', '식', '영', '웅', '음', '전', '파']

    # SEP: Start, End, Padding
    vocab = ''.join(eng + kor) + 'SEP'
    print(vocab)

    char2idx = {c: i for i, c in enumerate(vocab)}
    print(char2idx)     # {'a': 0, 'b': 1, 'd': 2, ...}

    return vocab, char2idx


# (인코더 입력, 디코더 입력, 디코더 출력) 데이터 생성
def make_batch(data, char2idx):
    onehots = np.eye(len(char2idx), dtype=np.float32)

    enc_inputs, dec_inputs, dec_target = [], [], []
    for eng, kor in data:
        print(eng, 'S'+kor, kor+'E')

        enc_in = [char2idx[c] for c in eng]
        dec_in = [char2idx[c] for c in 'S'+kor]
        target = [char2idx[c] for c in kor+'E']

        print(enc_in, dec_in, target)
        print()

        enc_inputs.append(onehots[enc_in])
        dec_inputs.append(onehots[dec_in])
        dec_target.append(target)           # sparse

    return np.float32(enc_inputs), np.float32(dec_inputs), np.float32(dec_target)


def show_seq2seq(enc_inputs, dec_inputs, dec_target, vocab):
    n_classes, n_hiddens = len(vocab), 128

    ph_enc_in = tf.placeholder(tf.float32, [None, None, n_classes]) # (batch_size, time_steps, n_classes)
    ph_dec_in = tf.placeholder(tf.float32, [None, None, n_classes])

    enc_cell = tf.nn.rnn_cell.BasicRNNCell(n_hiddens, name='enc_cell')
    _, enc_states = tf.nn.dynamic_rnn(enc_cell, ph_enc_in, dtype=tf.float32)

    dec_cell = tf.nn.rnn_cell.BasicRNNCell(n_hiddens, name='dec_cell')
    dec_output, _ = tf.nn.dynamic_rnn(dec_cell, ph_dec_in, dtype=tf.float32, initial_state=enc_states)

    z = tf.layers.dense(dec_output, n_classes, activation=None)
    hx = tf.nn.softmax(z)

    # ------------------------------------ #
    # 6-1 파일에서 복사해 옴

    dec_target = np.int32(dec_target)
    loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=dec_target, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.AdamOptimizer(0.001)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train, {ph_enc_in: enc_inputs, ph_dec_in: dec_inputs})
        print(i, sess.run(loss, {ph_enc_in: enc_inputs, ph_dec_in: dec_inputs}))

    sess.close()


data = [('food', '음식'), ('wood', '나무'),
        ('blue', '파랑'), ('lamp', '전구'),
        ('wind', '바람'), ('hero', '영웅')]

vocab, char2idx = make_vocab_and_index(data)
enc_inputs, dec_inputs, dec_target = make_batch(data, char2idx)
print(enc_inputs.shape, dec_inputs.shape, dec_target.shape)
# (6, 4, 30) (6, 3, 30) (6, 3)

show_seq2seq(enc_inputs, dec_inputs, dec_target, vocab)

