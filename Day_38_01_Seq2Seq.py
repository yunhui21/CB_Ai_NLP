# Day37_02_Seq2Seq_word.py
# Day_37_01_seq2seq.py
import re
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# 문제
# 새롭게 제공된 문장을 사용해서 학습
# 기계번역

def make_vocab_and_index(data):
    # 문제
    # 영어 알파벳과 한글 글자로 분리하세요
    # 토크나이저를 사용해서
    eng = sorted({c for w, _ in data for c in w.split()}) # <w.split()>구두점이 더 많으면 토크나이저를 사용하는것이 좋다.
    kor = sorted({c for _, w in data for c in w.split()})

    # print(eng)  # ['are', 'blue', 'christmas', 'color', 'did', 'do', "don't", 'eat', 'food',
    # print(kor)  # ['거기에는', '것일까', '나는', '나무가', '너는', '머었니', '바람은', '않아', '어디에
    # ' ', "'", '?'제거

    # SEP: Start, End, Padding
    vocab = eng + kor + ['_sos_', '_eos_', '_pad_']
    # print(vocab)
    '''['are', 'blue', 'christmas', 'color', 'did', 'do', "don't", 'eat', 'food', 
    'forest', 'hero', 'how', 'i', 'is', 'lamp', 'like', 'many', 'need', 'pretty', 
    'reality', 'so', 'some', 'the', 'there', 'trees', 'we', 'where', 'wind', 'you', 
    '거기에는', '것일까', '나는', '나무가', '너는', '머었니', '바람은', '않아', '어디에서는', 
    '얼마나', '영웅은', '예뻐', '우리에게', '음식', '있어?', '전구는', '정말', '좀', '좋아해',
     '진짜', '크리스마스', '파랑색을', '필요하지', '_sos_', '_eos_', '_pad_']'''
    char2idx = {c: i for i, c in enumerate(vocab)}
    # print(char2idx)
    '''
    {'are': 0, 'blue': 1, 'christmas': 2, 'color': 3, 'did': 4, 'do': 5, 
    "don't": 6, 'eat': 7, 'food': 8, 'forest': 9, 'hero': 10, 'how': 11, 
    'i': 12, 'is': 13, 'lamp': 14, 'like': 15, 'many': 16, 'need': 17, 
    'pretty': 18, 'reality': 19, 'so': 20, 'some': 21, 'the': 22, 'there': 23, 
    'trees': 24, 'we': 25, 'where': 26, 'wind': 27, 'you': 28, '거기에는': 29, 
    '것일까': 30, '나는': 31, '나무가': 32, '너는': 33, '머었니': 34, '바람은': 35, 
    '않아': 36, '어디에서는': 37, '얼마나': 38, '영웅은': 39, '예뻐': 40, '우리에게': 41, 
    '음식': 42, '있어?': 43, '전구는': 44, '정말': 45, '좀': 46, '좋아해': 47, '진짜': 48, '크리스마스': 49, 
    '파랑색을': 50, '필요하지': 51, '_sos_': 52, '_eos_': 53, '_pad_': 54}'''
    return vocab, char2idx
    # {' ': 22, "'": 1, 'a': 2, 'b': 3, 'c': 4, 'd': 5, 'e': 6, 'f': 7,

# def clean_str(string, TREC=True):
#     string = re.sub(r"[^가-힣A-Za-z0-9(),!?\'\'] ", " ", string)
#     string = re.sub(r"\'s", " \'s", string)
#     string = re.sub(r"\'ve", " \'ve", string)
#     string = re.sub(r"n\'t", " n\'t", string)
#     string = re.sub(r"\'re", " \'re", string)
#     string = re.sub(r"\'d", " \'d", string)
#     string = re.sub(r"\'ll", " \'ll", string)
#     string = re.sub(r",", " , ", string)
#     string = re.sub(r"!", " ! ", string)
#     string = re.sub(r"\(", " \( ", string)
#     string = re.sub(r"\)", " \) ", string)
#     string = re.sub(r"\?", " \? ", string)
#     string = re.sub(r"\s{2,}", " ", string)
#     return string.strip() if TREC else string.strip().lower()


# (인코더 입력, 디코더 입력, 디코더 출력) 데이터 생성
def make_batch(data, char2idx):
    onehots = np.eye(len(char2idx), dtype=np.float32)

    enc_inputs, dec_inputs, dec_target = [], [], []
    for eng, kor in data:
        print(eng, '_SOS_ '+kor, kor+' _EOS_')

        enc_in = [char2idx[c] for c in eng.split()]
        dec_in = [char2idx[c] for c in ('_SOS_ '+kor).split()]
        target = [char2idx[c] for c in (kor+' _EOS_').split()]

        print(enc_in, dec_in, target)
        print()

        enc_inputs.append(onehots[enc_in])
        dec_inputs.append(onehots[dec_in])
        dec_target.append(target)           # sparse

    return np.float32(enc_inputs), np.float32(dec_inputs), np.float32(dec_target)



def show_seq2seq(enc_inputs, dec_inputs, dec_target, vocab, char2idx):
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

    # 문제
    # 잘예측하는지 출력해주세요.(hero에 대해서만 검증하세요)
    # dec_in을 만들어주어야 한다.
    new_data=[('hero', 'pp'), ('blue', 'pp')]
    enc_inputs, dec_inputs, _ = make_batch(new_data, char2idx)

    preds =sess.run(hx, {ph_enc_in: enc_inputs, ph_dec_in: dec_inputs})
    print(preds)
    print(preds.shape) # (2, 3, 30)

    preds_arg = np.argmax(preds, axis=2)
    print(preds_arg) # [[22 23 28] [26 18 28]]

    # results = [[vocab[j] for j in i] for i in preds_arg]
    # print(results) # [['영', '웅', 'E'], ['파', '랑', 'E']]
    results = [[vocab[j] for j in i[:-1]] for i in preds_arg]
    print(results)  # [['영', '웅', 'E'], ['파', '랑', 'E']]
    print([''.join(i) for i in results])
    sess.close()


# data = [('did you eat some food', '너는 음식 좀 머었니'),
#         ('how many trees are there', '거기에는 나무가 얼마나 있어?'),
#         ('i reality like blue color', '나는 파랑색을 진짜 좋아해'),
#         ('christmas lamp is so pretty', '크리스마스 전구는 정말 예뻐'),
#         ('where do wind some forest', '바람은 어디에서는 것일까'),
#         ("we don't need the hero", '우리에게 영웅은 필요하지 않아')]


vocab, char2idx = make_vocab_and_index(data)
enc_inputs, dec_inputs, dec_target = make_batch(data, char2idx)
print(enc_inputs.shape, dec_inputs.shape, dec_target.shape)
#
#
# show_seq2seq(enc_inputs, dec_inputs, dec_target, vocab, char2idx)
#
