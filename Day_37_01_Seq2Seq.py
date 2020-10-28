# # Day_37_01_Seq2Seq.py
# import numpy as np
# import tensorflow.compat.v1 as tf
# # print(tf.__version__)
# tf.disable_v2_behavior()
#
#
# # 문제
# # 영어 알파벳과 한글 글자로 모아주세요.
# def make_vocab_and_index(data):
#     data = [('food', '음식'), ('wood', '나무'),
#             ('blue', '파랑'), ('lamp', '전구'),
#             ('wind', '바람'), ('hero', '영웅')]
#
#     eng = sorted({w for w, _ in data for c in w})
#     kor = sorted({w for _, w in data for c in w})
#
#     print(eng)
#     print(kor)
#
#
#
# # 데이터의 길이가 다를때는 padding을 넣어서 길이를 동일하게 한다..
# make_vocab_and_index(data)

# Day_37_01_seq2seq.py
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()



# def make_vocab_and_index(data):
#     # 문제
#     # 영어 알파벳과 한글 글자로 분리하세요
#     eng = sorted({c for w, _ in data for c in w})
#     kor = sorted({c for _, w in data for c in w})
#
#     # print(eng) # ['a', 'b', 'd', 'e', 'f', 'h', 'i', 'l', 'm', 'n', 'o', 'p', 'r', 'u', 'w']
#     # print(kor) # ['구', '나', '람', '랑', '무', '바', '식', '영', '웅', '음', '전', '파']
#
#     # start, end, padding 실전에서는 paddingdl 필요
#     vocab = ''.join(eng+kor) + 'SEP'
#     # print(vocab) # a b d e f h i l m n o p r u w 구 나 람 랑 무 바 식 영 웅 음 전 파SEP
#
#     char2idx = {c: i for i, c in enumerate(vocab)}
#     # print(char2idx)
#     '''
#     {'a': 0, ' ': 51, 'b': 2, 'd': 4, 'e': 6, 'f': 8, 'h': 10, 'i': 12,
#     'l': 14, 'm': 16, 'n': 18, 'o': 20, 'p': 22, 'r': 24, 'u': 26, 'w': 28,
#     '구': 30, '나': 32, '람': 34, '랑': 36, '무': 38, '바': 40, '식': 42,
#     '영': 44, '웅': 46, '음': 48, '전': 50, '파': 52, 'S': 53, 'E': 54, 'P': 55}
#     '''
#     return vocab, char2idx
# encoder입력, decoder입력, decoder출력
#
# def make_batch(data, char2idx):
#
#     # onehot벡터로 3차원으로 변화해준다.
#     onehot = np.eye(len(char2idx), dtype=np.float32)
#     enc_inputs, dec_inputs, dec_target = [], [], []
#     for eng, kor in data:
#         print(eng, 'S'+kor, kor+'E')
#
#         enc_in = [char2idx[c] for c in eng]
#         dec_in = [char2idx[c] for c in 'S'+kor]
#         target = [char2idx[c] for c in kor+'E']
#
#         print(enc_in,dec_in, target)
#         print()
#
#         enc_inputs.append(onehot[enc_in])
#         dec_inputs.append(onehot[dec_in])
#         dec_target.append(target)       # sparse
#
#     return np.float32(enc_inputs), np.float32(dec_inputs), np.float32(dec_target)
#
# def show_seq2seq(enc_inputs, dec_inputs, dec_target, vocab):
#
#     n_classes, n_ = len(vocab), 128
#     ph_enc_in = tf.placeholder(tf.float32, [None, None, n_classes])
#     ph_dec_in = tf.placeholder(tf.float32, [None, None, n_classes])
#
#
#
# data = [('food', '음식'), ('wood', '나무'),
#         ('blue', '파랑'), ('lamp', '전구'),
#         ('wind', '바람'), ('hero', '영웅')]
#
# vocab, char2idx = make_vocab_and_index(data)
# enc_inputs, dec_inputs, dec_target = make_batch(data, char2idx)
# print(enc_inputs.shape, dec_inputs.shape, dec_target.shape)
# (6, 4, 30) (6, 3, 30) (6, 3)
# show_seq2seq(enc_inputs, dec_inputs, dec_target, vocab)
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
        dec_target.append(target)

    return np.float32(enc_inputs), np.float32(dec_inputs), np.float32(dec_target)

def show_seq2seq(enc_inputs, dec_inputs, dec_target, vocab):

    n_classes, n_ = len(vocab), 128
    ph_enc_in = tf.placeholder(tf.float32, [None, None, n_classes])
    ph_dec_in = tf.placeholder(tf.float32, [None, None, n_classes])

    enc_cell = tf.nn.rnn_call.BasicRNNCell(n_hiddens, name='enc_cell')
    _, enc_states = tf.nn.dynamic_rnn(enc_cell, ph_enc_in, dtype=tf.float32) # context

    dec_cell = tf.nn.rnn_call.BasicRNNCell(n_hiddens, name='dec_cell')
    dec_output, _ = tf.nn.dynamic_rnn(dec_cell, ph_dec_in, dtype=tf.float32, initial_state = enc_states) # context

    z = tf.layers.dense(dec_output, n_classes, activation=None)
    hx = tf.nn.softmax(z)

    #---------------------------6-1, copy-------#

    loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=dec_target, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.AdanOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train, {ph_enc_in: enc_inputs, ph_dec_in:dec_inputs})
        print(i, sess.run(loss,  {ph_enc_in: enc_inputs, ph_dec_in:dec_inputs}))
    sess.close()




data = [('food', '음식'), ('wood', '나무'),
        ('blue', '파랑'), ('lamp', '전구'),
        ('wind', '바람'), ('hero', '영웅')]


vocab, char2idx = make_vocab_and_index(data)
enc_inputs, dec_inputs, dec_target = make_batch(data, char2idx)
print(enc_inputs.shape, dec_inputs.shape, dec_target.shape)
# (6, 4, 30) (6, 3, 30) (6, 3)
show_seq2seq(enc_inputs, dec_inputs, dec_target, vocab)