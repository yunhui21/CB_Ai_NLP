# Day_06_04_numpy.py
import numpy as np


def slicing():
    a = [0, 1, 2, 3, 4, 5, 6]
    print(a[0], a[1])
    print(a[len(a)-1], a[len(a)-2])
    print(a[-1], a[-2])

    print(a[0:3])       # slicing, range
    print(a[:3])

    # 문제
    # 뒤쪽 나머지를 출력하세요
    print(a[3:7])
    print(a[3:])

    # 문제
    # 짝수 번째만 출력하세요
    # 홀수 번째만 출력하세요
    print(list(range(0, 7, 2)))
    print(a[0:7:2])
    print(a[::2])

    print(list(range(1, 7, 2)))
    print(a[1:7:2])
    print(a[1::2])
    print()

    # 문제
    # 리스트를 거꾸로 출력하세요
    print(a[3])
    print(a[3:4])
    print(a[3:3])
    print(a[6:0:-1])
    print(a[6:-1:-1])
    print(a[-1:-1:-1])

    print(a[::-1])      # 증감(양수: 정방향, 음수: 역방향)

    # 문제
    # 거꾸로 짝수 번째를 출력하세요
    # 거꾸로 홀수 번째를 출력하세요
    print(a[::-2])
    print(a[-2::-2])


def np_basic():
    a = np.arange(12)
    print(a)
    print(type(a))
    print(a.shape, a.dtype, a.size)

    # b = a.reshape(3, 4)
    # b = a.reshape(3, -1)
    b = a.reshape(-1, 4)
    print(b)
    print(type(b))
    print(b.shape, b.dtype, b.size)

    # 문제
    # 2차원 배열을 1차원으로 바꾸세요 (3가지)
    print(b.reshape(12))
    print(b.reshape(b.size))
    print(b.reshape(b.shape[0] * b.shape[1]))
    print(b.reshape(-1))
    # print(b.reshape(1, -1))
    # print(b.reshape(1, -1).shape)
    print('-' * 30)

    print(a)

    print(a + 1)                # broadcast
    print(a ** 2)
    print(2 ** a)
    print(a > 5)

    t = (a > 5)
    print(t)
    print(a[t])
    print(a[a > 5])

    print('-' * 30)

    print(b + 1)                # broadcast
    print(b ** 2)
    print(2 ** b)
    print(b > 5)

    print('-' * 30)

    print(a + a)                # vector
    print(b + b)

    print('-' * 30)

    print(np.sin(a))            # universal
    print(np.sin(b))

    print('\n\n\n')











# slicing()
np_basic()

