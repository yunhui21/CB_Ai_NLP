# Day_27_03_JenaClimateRnn.py
import csv
import tensorflow as tf
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt


def get_jena():
    f = open('data/jena_climate_2009_2016.csv', 'r', encoding='utf-8')
    f.readline()

    jena = []
    for row in csv.reader(f):
        # print(row[1:])
        # break

        jena.append([float(i) for i in row[1:]])

    f.close()

    # print(*jena[:3], sep='\n')
    return np.float32(jena)


def generator_basic():
    for i in range(5):
        print(i)


    def simple_generator():
        yield 'a'
        yield 'b'
        yield 'c'


    for i in simple_generator():
        print(i)

    simple = simple_generator()

    i0 = next(simple)
    print(i0)

    i1 = next(simple)
    print(i1)

    i2 = next(simple)
    print(i2)

    try:
        i3 = next(simple)
    except StopIteration:
        print('StopIteration')

    print('-' * 30)


    def real_generator():
        for i in range(5):
            yield i * 100


    for j in real_generator():
        print(j)


    # 문제
    # 100보다 작은 난수 5개씩 반환하는 횟수 무제한의 제너레이터를 만드세요
    def random_generator():
        while True:
            numbers = []
            for i in range(5):
                numbers.append(random.randrange(100))
            yield numbers


    for i, j in enumerate(random_generator()):
        print(j)

        if i >= 5:
            break


def jena_generator(rows, lookback, delay, batch_size, step, min_idx, max_idx, shuffle):
    i = min_idx + lookback
    while True:
        if shuffle:
            pos = np.random.randint(min_idx + lookback, max_idx, size=batch_size)
        else:
            if i + batch_size > max_idx:
                i = min_idx + lookback
            pos = np.arange(i, i + batch_size)
            i += batch_size

        samples = np.zeros([batch_size, lookback // step, rows.shape[-1]])
        targets = np.zeros([batch_size])

        for j, row in enumerate(pos):
            indices = range(row - lookback, row, step)
            # samples[j] = rows[[1, 3, 5]]
            samples[j] = rows[indices]
            targets[j] = rows[row + delay][1]       # degC. 날짜 인덱스는 제거한 상태

        yield samples, targets


def make_generator():
    rows = get_jena()

    # 학습(20만), 검증(10만), 검사(11만)
    mean = rows[:200000].mean(axis=0)
    rows -= mean
    std = rows[:200000].std(axis=0)
    rows /= std

    # 참고할 기간, 예측하려고 하는 시점, 배치 크기, 참조 간격
    # 입력 갯수    타겟                샘플 갯수  시간당 1개
    per_day = (60 * 24) // 10   # 144 = 6 * 24
    lookback, delay, batch_size, step = per_day * 10, per_day, 128, 6
    n_features = rows.shape[-1]

    steps_valid = (   300000 - 200000 - lookback) // batch_size
    steps_test  = (len(rows) - 300000 - lookback) // batch_size

    train_gen = jena_generator(rows, lookback, delay, batch_size, step, 0, 200000, True)
    valid_gen = jena_generator(rows, lookback, delay, batch_size, step, 200000, 300000, False)
    test_gen  = jena_generator(rows, lookback, delay, batch_size, step, 300000, len(rows), False)

    # test_gen, steps_test은 여기서는 사용하지 않음
    return train_gen, valid_gen, steps_valid, std, n_features


def save_history(history, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(history.history, f)


def show_history(file_path, title):
    with open(file_path, 'rb') as f:
        history = pickle.load(f)
        # print(history.keys())

        x = range(len(history['loss']))
        plt.plot(x, history['loss'], 'r', label='train')
        plt.plot(x, history['val_loss'], 'g', label='valid')
        plt.title(title)
        plt.legend()
        plt.show()


def model_baseline(valid_gen, steps_valid, std):
    batch = []
    for step in range(steps_valid):
        samples, targets = next(valid_gen)
        # print(samples.shape, targets.shape)

        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch.append(mae)

    mean = np.mean(batch)
    print('mae mean :', mean)
    print('celcius  :', mean * std[1])      # 1: degC


def model_fc(train_gen, valid_gen, steps_valid, n_features):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer='rmsprop', loss='mae')
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=500,
                                  epochs=20,
                                  validation_data=valid_gen,
                                  validation_steps=steps_valid,
                                  verbose=2)
    # model.summary()

    save_history(history, 'data/jena_1_fc.pickle.history')


def model_gru(train_gen, valid_gen, steps_valid, n_features):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input([None, n_features]))
    model.add(tf.keras.layers.GRU(32))
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer='rmsprop', loss='mae')
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=500,
                                  epochs=20,
                                  validation_data=valid_gen,
                                  validation_steps=steps_valid,
                                  verbose=2)
    save_history(history, 'data/jena_2_gru.pickle.history')


def model_gru_dropout(train_gen, valid_gen, steps_valid, n_features):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input([None, n_features]))

    model.add(tf.keras.layers.GRU(32, dropout=0.2, recurrent_dropout=0.2)) # dropout mask , dropout이 2개인 이유, input - dropout, state - recurrent_dropout

    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer='rmsprop', loss='mae')
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=500,
                                  epochs=40,
                                  validation_data=valid_gen,
                                  validation_steps=steps_valid,
                                  verbose=2)
    save_history(history, 'data/jena_3_gru_dropout.pickle.history')


def model_gru_stack(train_gen, valid_gen, steps_valid, n_features):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input([None, n_features]))

    model.add(tf.keras.layers.GRU(32, dropout=0.1,recurrent_dropout=0.5, return_sequences=True))  # dropout mask , dropout이 2개인 이유, input - dropout, state - recurrent_dropout
    model.add(tf.keras.layers.GRU(64, dropout=0.1,recurrent_dropout=0.5))
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer='rmsprop', loss='mae')
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=500,
                                  epochs=40,
                                  validation_data=valid_gen,
                                  validation_steps=steps_valid,
                                  verbose=2)
    save_history(history, 'data/jena_4_gru_stack.pickle.history')


def model_bidirectional(train_gen, valid_gen, steps_valid, n_features):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input([None, n_features]))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)))
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer='rmsprop', loss='mae')
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=500,
                                  epochs=40,
                                  validation_data=valid_gen,
                                  validation_steps=steps_valid,
                                  verbose=2)
    save_history(history, 'data/jena_5_bidirectional.pickle.history')


def model_1d_conv(train_gen, valid_gen, steps_valid, n_features):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input([None, n_features]))
    model.add(tf.keras.layers.Conv1D(32, 5, activation='relu'))
    model.add(tf.keras.layers.MaxPool1D(3))
    model.add(tf.keras.layers.Conv1D(32, 5, activation='relu'))
    model.add(tf.keras.layers.MaxPool1D(3))
    model.add(tf.keras.layers.Conv1D(32, 5, activation='relu'))
    model.add(tf.keras.layers.MaxPool1D(3))
    model.add(tf.keras.layers.GlobalAveragePooling1D())
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer='rmsprop', loss='mae')
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=500,
                                  epochs=20,
                                  validation_data=valid_gen,
                                  validation_steps=steps_valid,
                                  verbose=2)
    save_history(history, 'data/jena_6_1d_conv.pickle.history')


def model_1d_conv_rnn(train_gen, valid_gen, steps_valid, n_features):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input([None, n_features]))
    model.add(tf.keras.layers.Conv1D(32, 5, activation='relu'))
    model.add(tf.keras.layers.MaxPool1D(3))
    model.add(tf.keras.layers.GRU(32, dropout=0.1, recurrent_dropout=0.5))
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer='rmsprop', loss='mae')
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=500,
                                  epochs=20,
                                  validation_data=valid_gen,
                                  validation_steps=steps_valid,
                                  verbose=2)
    save_history(history, 'data/jena_7_1d_conv_rnn.pickle.history')

# generator_basic()

train_gen, valid_gen, steps_valid, std, n_features = make_generator()

# model_baseline(valid_gen, steps_valid, std)
# model_fc(train_gen, valid_gen, steps_valid, n_features)
# model_gru(train_gen, valid_gen, steps_valid, n_features)
# model_gru_dropout(train_gen, valid_gen, steps_valid, n_features)
# model_gru_stack(train_gen, valid_gen, steps_valid, n_features)
# model_bidirectional(train_gen,valid_gen,steps_valid,n_features)
# model_1d_conv(train_gen,valid_gen,steps_valid,n_features)
# model_1d_conv_rnn(train_gen,valid_gen,steps_valid,n_features)

# show_history('data/jena_1_fc.pickle', '1_fc')
# show_history('data/jena_2_gru.pickle', '2_gru')
# show_history('data/jena_3_gru_dropout.pickle', '3_dropout')
# show_history('data/jena_4_gru_stack.pickle', '4_stack')
# show_history('data/jena_5_bidirectional.pickle', '5_bidirectional')
# show_history('data/jena_6_1d_conv.pickle', '6_id_conv')
show_history('data/jena_7_1d_conv_rnn.pickle', '7_id_conv_rnn')


# baseline
# mae mean : 0.2895216161408672
# celcius  : 2.562940948382028

# 성능향상
# 1.
# 2.
# 3. 각각의 모델의 구현된 기법들을 합성
# 4. 최종 모델로부터 최적의 순간 포착


# a b c d e f g
#     c d e
# a b c
#   b c d

# a b c
# d e f
# g h i
# j k l
# m n o
# p q r

# a b c
# d e f
# g h i

# d e f
# g h i
# j k l

# g h i
# j k l
# m n o

# j k l
# m n o
# p q r
