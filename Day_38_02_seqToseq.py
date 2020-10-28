# Day_38_02_seqToseq.py
# 케라스버전으로
# Day_37_01_seq2seq.py
import numpy as np
import tensorflow as tf
# import tensorflow.compat.v1 as tf # 텐서플로 코드
# tf.disable_v2_behavior()

# 문제
# show함수를 케라스버전으로 바꾸세요
# simplelayers를 사용.
# state를 반환하려면 return state 옵션을 사용해야 합니다.반환값이 2개
# 함수형을 사용. 2개의 입력을 처리하기 위해서.

# 원본파일 수정하지 않는 함수.
def make_vocab_and_index(data):
    # 문제
    # 영어 알파벳과 한글 글자로 분리하세요
    eng = sorted({c for w, _ in data for c in w})
    kor = sorted({c for _, w in data for c in w})

    # print(eng)  # ['a', 'b', 'd', 'e', 'f', 'h', 'i', 'l', 'm', 'n', 'o', 'p', 'r', 'u', 'w']
    # print(kor)  # ['구', '나', '람', '랑', '무', '바', '식', '영', '웅', '음', '전', '파']

    # SEP: Start, End, Padding
    vocab = ''.join(eng + kor) + 'SEP'
    # print(vocab)

    char2idx = {c: i for i, c in enumerate(vocab)}
    # print(char2idx)     # {'a': 0, 'b': 1, 'd': 2, ...}

    return vocab, char2idx

# 원본파일 수정하지 않는 함수.
def make_batch(data, char2idx):
    onehots = np.eye(len(char2idx), dtype=np.float32)

    enc_inputs, dec_inputs, dec_target = [], [], []
    for eng, kor in data:
        # print(eng, 'S'+kor, kor+'E')

        enc_in = [char2idx[c] for c in eng]
        dec_in = [char2idx[c] for c in 'S'+kor]
        target = [char2idx[c] for c in kor+'E']

        # print(enc_in, dec_in, target)
        # print()

        enc_inputs.append(onehots[enc_in])
        dec_inputs.append(onehots[dec_in])
        dec_target.append(target)           # sparse

    return np.float32(enc_inputs), np.float32(dec_inputs), np.float32(dec_target)

# 수정이 필요한 함수 : 텐서플로 버전을 케라스 버전으로 변환
def show_seq2seq(enc_inputs, dec_inputs, dec_target, vocab, char2idx):
    n_classes, n_hiddens = len(vocab), 128

    enc_inputs_layer = tf.keras.layers.Input([None, n_classes])
    _, enc_states = tf.keras.layers.SimpleRNN(n_hiddens, return_state=True)(enc_inputs_layer)

    # [<tf.Tensor 'simple_rnn/strided_slice_3:0' shape=(None, 128) dtype=float32>: OUTPUT,
    # <tf.Tensor 'simple_rnn/while:4' shape=(None, 128) dtype=float32>] : STATE
    # print(a(enc_inputs_layer))# placeholder 역할

    dec_inputs_layer = tf.keras.layers.Input([None, n_classes])
    dec_output = tf.keras.layers.SimpleRNN(n_hiddens,
                                           return_sequences=True)(dec_inputs_layer,
                                                                  initial_state=enc_states)
    # sequence True는 전체를 다 리턴한다.
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(dec_output)


    model = tf.keras.Model([enc_inputs_layer, dec_inputs_layer], outputs)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.sparse_categorical_crossentropy)

    model.fit([enc_inputs, dec_inputs], dec_target, epochs=100)

    # 문제
    # 잘 예측하는지 결과를 출력하세요 (hero에 대해서만 검증하세요)
    new_data = [('hero', 'PP'), ('blue', 'PP')]
    enc_inputs, dec_inputs, _ = make_batch(new_data, char2idx)

    preds = model.predict([enc_inputs, dec_inputs])
    preds_arg = np.argmax(preds, axis=2)

    results = [[vocab[j] for j in i[:-1]] for i in preds_arg]
    print([''.join(i) for i in results])  # ['영웅', '파랑']



data = [('food', '음식'), ('wood', '나무'),
        ('blue', '파랑'), ('lamp', '전구'),
        ('wind', '바람'), ('hero', '영웅')]

vocab, char2idx = make_vocab_and_index(data)
enc_inputs, dec_inputs, dec_target = make_batch(data, char2idx)
print(enc_inputs.shape, dec_inputs.shape, dec_target.shape)
# (6, 4, 30) (6, 3, 30) (6, 3)
show_seq2seq(enc_inputs, dec_inputs, dec_target, vocab, char2idx)


