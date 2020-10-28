# Day_37_02_seq2seq_word.py
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# 문제
# 새롭게 제공된 영어와 한글 문장에 대해 학습하고 예측하세요


def make_vocab_and_index(data):
    # 문제
    # 영어 알파벳과 한글 단어로 분리하세요
    eng = sorted({c for w, _ in data for c in w.split()})
    kor = sorted({c for _, w in data for c in w.split()})

    print(eng)  # ['are', 'blue', 'christmas', ...]
    print(kor)  # ['거기에는', '것일까', '나는', ...]

    # SEP: Start, End, Padding
    vocab = eng + kor + ['_SOS_', '_EOS_', '_PAD_']
    print(vocab)

    char2idx = {c: i for i, c in enumerate(vocab)}
    print(char2idx)     # {'are': 0, 'blue': 1, 'christmas': 2, ...}

    return vocab, char2idx


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
    # 잘 예측하는지 결과를 출력하세요 (hero에 대해서만 검증하세요)
    new_data = [('we dont need the hero', '_PAD_ _PAD_ _PAD_ _PAD_'),
                ('i really like blue color', '_PAD_ _PAD_ _PAD_ _PAD_')]
    enc_inputs, dec_inputs, _ = make_batch(new_data, char2idx)

    preds = sess.run(hx, {ph_enc_in: enc_inputs, ph_dec_in: dec_inputs})
    # print(preds)
    print(preds.shape)      # (2, 5, 57)

    preds_arg = np.argmax(preds, axis=2)
    print(preds_arg)        # [[44 41 53 38 55] [32 52 50 49 55]]

    results = [[vocab[j] for j in i] for i in preds_arg]
    print(results)          # [['우리에게', '영웅은', '필요하지', '않아', '_EOS_'], ['나는', '파랑색을', '진짜', '좋아해', '_EOS_']]

    results = [[vocab[j] for j in i[:-1]] for i in preds_arg]
    print(results)          # [['우리에게', '영웅은', '필요하지', '않아'], ['나는', '파랑색을', '진짜', '좋아해']]
    print([''.join(i) for i in results])    # ['우리에게영웅은필요하지않아', '나는파랑색을진짜좋아해']

    sess.close()


data = [('did you eat some food', '너는 음식을 좀 먹었니'),
        ('how many trees are there', '거기에는 나무가 얼마나 있니'),
        ('i really like blue color', '나는 파랑색을 진짜 좋아해'),
        ('christmas lamp is so pretty', '크리스마스 전구가 너무 예뻐'),
        ('where do wind come from', '바람은 어디에서 오는 것일까'),
        ('we dont need the hero', '우리에게 영웅은 필요하지 않아')]
# def convert(4);
#   return ''.
# data = [(list('food'), list('음식')), (list('wood'), list('나무')),
#         (list('blue'), list('파랑')), (list('lamp'), list('전구')),
#         (list('wind'), list('바람')), (list('hero'), list('영웅'))]

vocab, char2idx = make_vocab_and_index(data)
enc_inputs, dec_inputs, dec_target = make_batch(data, char2idx)
print(enc_inputs.shape, dec_inputs.shape, dec_target.shape)
# (6, 5, 57) (6, 5, 57) (6, 5)

show_seq2seq(enc_inputs, dec_inputs, dec_target, vocab, char2idx)
