# Day_11_01_KerasBasic.py

import tensorflow as tf
import csv

#  문제
# 속도가 30과 50일대의 제동 거리를 구하세요.
# (csv 파일을 읽을 때, csv모듈을 사용해서 읽어봅시다.)
# 기본예제들을 케라스로 바꾸는 작업 :

# 문제
# girth가 10이고 height가 70일대
# girth가 20이고 height가 80일때의 volume을 예측하세요.

def linear_basic():
    x = [1, 2, 3]
    y = [1, 2, 3]

    #모델을 만드는 첫번째 방식

    model = tf.keras.Sequential()   #세션을 구축했다.
    model.add(tf.keras.layers.Dense(1, input_dim=1))

    model.compile(optimizer='sgd', loss = 'mse')
    model.fit(x, y, epochs=100)  #keras
    #예측방법: 결과만보기, 일일이 확인하는범 2개지
    print(model.evaluate(x, y))

    preds = model.predict(x)
    print(preds)

    model.summary() #wx + b : trainable params w = 2개


def linear_cars():
    x, y = [], []
    f = open('data/cars.csv', 'r', encoding='utf-8')
    f.readline()

    for row in csv.reader(f):
        # print(row)
        x.append(int(row[1]))
        y.append(int(row[2]))
    # rows = [row.strip().split(',') for row in f]
    f.close()
#-----------------------------------------------------#

    # 모델을 만드는 첫번째 방식
    model = tf.keras.Sequential()  # 세션을 구축했다.
    model.add(tf.keras.layers.Dense(1, input_dim=1))

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001),
                  loss=tf.keras.losses.mse)
    # loss 에 들어간건 함수 , 호출된 함수를 사용하는 형태
    # optimizer = 클래스 객체

    # model.fit(x, y, epochs=100, verbose=0)  # 32/50 [==================>...........] - ETA: 0s - loss: 187.2625
                                              # 50/50 [==============================] - 0s 438us/sample - loss: 259.7535
    # model.fit(x, y, epochs=100, verbose=1)  # 풀출력
    model.fit(x, y, epochs=100, verbose=2)    # 요약해서 출력

    # 예측방법: 결과만보기, 일일이 확인하는범 2가지
    print(model.evaluate(x, y))

    # preds = model.predict(x)
    preds = model.predict([30, 50])
    print(preds)

    model.summary()  # wx + b : trainable params w = 2개

def multifule_trees():
    x, y = [], []
    f = open('data/trees.csv', 'r', encoding='utf-8')
    f.readline()

    for row in csv.reader(f):
        # print(row)
        x.append(int(row[1]))
        y.append(int(row[3]))
    f.close()
#-----------------------------------------------------#

    # 모델을 만드는 첫번째 방식
    model = tf.keras.Sequential()  # 세션을 구축했다.
    model.add(tf.keras.layers.Dense(1, input_dim=1))

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001),
                  loss=tf.keras.losses.mse)
    # loss 에 들어간건 함수 , 호출된 함수를 사용하는 형태
    # optimizer = 클래스 객체

    # model.fit(x, y, epochs=100, verbose=0)  # 32/50 [==================>...........] - ETA: 0s - loss: 187.2625
                                              # 50/50 [==============================] - 0s 438us/sample - loss: 259.7535
    # model.fit(x, y, epochs=100, verbose=1)  # 풀출력
    model.fit(x, y, epochs=100, verbose=2)    # 요약해서 출력

    # 예측방법: 결과만보기, 일일이 확인하는범 2가지
    print(model.evaluate(x, y))

    # preds = model.predict(x)
    preds = model.predict([10, 70])
    print(preds)

    model.summary()  # wx + b : trainable params w = 2개


# linear_basic()
# linear_cars()
multifule_trees()