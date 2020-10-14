# Day_27_03_Exem_3-3_JenaClimitRnn.py
import csv
import random
import pickle
import numpy as np
import tensorflow as tf

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

    # f = open('data/jena_climate_2009_2016.csv', 'r', encoding='utf-8')
    # f.readline()
    #
    # # degC 데이터만 갖고 오기
    # jena = []
    # for row in csv.reader(f):
    #     print(row[1:])# 0번재는 사용 안함. 시계열
    #     break
    #     jena.append([float(i) for i in row[1:]])
    # f.close()
    #
    # # print(jena[:3], sep='\n')
    # return np.float32(jena)

def generator_basic():
    for i in range(5):
        print(i)

    def simple_generator():
        # yield 0
        # yield 1
        # yield 2
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
        print(StopIteration)


    def real_generator():
        for i in range(5):
            yield i *100

    for j in real_generator():
        print(j)

    # 문제
    # 백보다 작은 난수 5개씩 반환하는 제너레이터를 만드세요.
    def random_generator():
        numbers = []
        for i in range(5):
            numbers.append(random.randrange(100))
        yield numbers

    for i, j in enumerate(random_generator()):
        print(j)

        if i>=5:
            break

# def jena_generator(rows, lookback, delay, batch_size, step, min_idx, max_idx, shuffle ):
#      i = min_idx + lookback
#      while True:
#         if shuffle:
#             pos = np.random.randint(min_idx * lookback, max_idx, size = batch_size)
#             # train에서 사용
#         else:
#             if i + batch_size > max_idx:
#                 i = min_idx + lookback # 한번더 사용할 수 있다.
#             pos = np.arange(i, i + batch_size) #
#             i += batch_size
#             # test, valid에서 사용
#             # 한번 호출마다 이 값을 갖고 옴.
#             samples = np.zeros([batch_size, lookback // step, rows.shape[-1]]) # 3차원 2차원에서 만들어낸 3차원
#             # samples = np.zeros([len(pos), lookback, rows.shape[-1]]) # 3차원
#             targets = np.zeros([batch_size]) # 1차원
#
#             for j, row in enumerate(pos): #
#                 indices = range(row - lookback, row, step) # 6개당 하나식 발췌 128개 row보다 작은값
#                 # samples[j] = rows[[1, 3, 5]] # 다시 2차원으로 값을 가져옴
#                 samples[j] = rows[indices] # 다시 2차원으로 값을 가져옴
#                 targets[j] = rows[row + delay][1] # row번째 값이며 jena에서 데이터를 1번째를 갖고옴  날짜나 인덱스를 제거한 상태
#
#             yield samples, targets

# def make_generator():
#     rows = get_jena()
#
#     # 학습(20만), 검증(10만), 검사(11만)
#     mean = rows[:200000].mean(axis=0)
#     rows -= mean
#     std = rows[:200000].std(axis=0)
#     rows /= std
#
#
#     # 참고할
#
#     per_day = (60 * 24) // 10
#     lookback ,delay , batch_size, step = per_day * 10, per_day, 128, 6
#     n_features = rows.shape[-1]
#     steps_valid = (200000-200000 * lookback) // batch_size
#     steps_test = (len(rows)-300000 * lookback) // batch_size
#
#
#     train_gen = jena_generator(rows, lookback, delay, batch_size, step, 0, 200000, True) # 시계열 데이터 순서를 shuffle해야한다.
#     valid_gen = jena_generator(rows, lookback, delay, batch_size, step, 200000, 300000, False)
#     test_gen = jena_generator(rows, lookback, delay, batch_size, step, 300000, len(rows), True)
#
#     # test_gen은 ㄱ
#     return train_gen, valid_gen, steps_valid, std, n_features
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


def show_history(file_path):
    with open(file_path, 'rb') as f:
        history = pickle.load(f)
        print(history.keys())


def model_baseline(valid_gen, steps_valid, std):
    batch = []
    for step in range(steps_valid):
        samples, targets = next(valid_gen)
        print(samples.shape, targets.shape)

        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds -targets))
        batch.append(mae)

    mean = np.mean(batch)
    print('.mean:', mean)
    print('celclus:', mean + std[1])

def model_fx(train_gen, valid_gen, steps_valid, n_features):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(1))

    # model.compile(optimizer ='rmsprop', loss='mse', metrics=['acc'])
    model.compile(optimizer ='rmsprop', loss='mae')
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=500,
                                  epochs=20,
                                  validation_data=valid_gen,
                                  validation_steps=steps_valid,
                                  verbose = 2) # batch_size그 크면 멈춘다. epoch가 한번이라도 돌면 문제는 없지만 컴퓨터 사양에 의해서 다운될수있다.

    # model.summary()
    save_history(history, 'data?/jena_model_1_fx.history')
# generator_basic()
train_gen, valid_gen, steps_valid, std , n_features= make_generator()
# model_baseline(valid_gen, steps_valid, std)
model_fx(train_gen, valid_gen, steps_valid, n_features)

# mae mean : 0.2895216161408672
# celcius  : 2.562940948382028

#  a b c d e f g h


#  a b c
#  d e f

#  a b c d e f g h
