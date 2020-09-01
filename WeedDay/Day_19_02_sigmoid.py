# Day_19_02_sigmoid.py
import numpy as np
import matplotlib.pyplot as plt


def show_sigmoid():
    def sigmoid(z):
        return 1 / (1 + np.e ** -z)

    print(np.e)
    print('-' * 30)

    print(sigmoid(-1))
    print(sigmoid(0))
    print(sigmoid(1))
    print('-' * 30)

    # 문제
    # -10 ~ 10 사이의 결과에 대해 시그모이드 그래프를 그리세요
    # for z in np.arange(-10, 10, 0.2):
    #     plt.plot(z, sigmoid(z), 'ro')
    # plt.show()

    z = np.arange(-10, 10, 0.2)
    s = sigmoid(z)
    plt.plot(z, s, 'ro')
    plt.show()


def show_logistic():
    def log_a():
        return 'A'

    def log_b():
        return 'B'

    y = 0
    print(y * log_a() + (1-y) * log_b())

    y = 1
    print(y * log_a() + (1-y) * log_b())


# show_sigmoid()
show_logistic()
