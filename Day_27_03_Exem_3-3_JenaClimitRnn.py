# Day_27_03_Exem_3-3_JenaClimitRnn.py
import csv
import random
import numpy as np
import tensorflow as tf

def get_jena():

    f = open('data/jena_climate_2009_2016.csv', 'r', encoding='utf-8')
    f.readline()

    # degC 데이터만 갖고 오기
    jena = []
    for row in csv.reader(f):
        print(row[1:])# 0번재는 사용 안함. 시계열
        break
        jena.append([float(i) for i in row[1:]])
    f.close()

    # print(jena[:3], sep='\n')
    return np.float32(jena)

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

generator_basic()