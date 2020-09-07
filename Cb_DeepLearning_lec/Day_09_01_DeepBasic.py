# Day_09_01_DeepBasic01.py
import numpy as np
import matplotlib.pyplot as plt


def cost(x, y, w):      # bias 생략
    t = 0
    for i in range(len(x)):
        hx = w * x[i]
        t += (hx - y[i]) ** 2

    return t / len(x)


def gradient_descent(x, y, w):      # bias 생략
    t = 0
    for i in range(len(x)):
        hx = w * x[i]
        t += (hx - y[i]) * x[i]

    return t / len(x)


def show_cost():
    #     1    0
    # hx= wx + b
    # y = x
    # y = 1 * x + 0
    # y = ax + b
    x = [1, 2, 3]
    y = [1, 2, 3]

    print(cost(x, y, 3))
    print(cost(x, y, 1))
    print()

    for i in range(-30, 50):
        w = i / 10
        c = cost(x, y, w)

        print(w, c)
        plt.plot(w, c, 'ro')

    plt.show()


# 문제
# w가 1이 될 수 있도록 코드를 수정하세요 (3가지)

# 문제
# x가 5와 7일 때의 y를 예측하세요
def show_gradient():
    x = [1, 2, 3]
    y = [1, 2, 3]

    w = 10
    for i in range(100):
        g = gradient_descent(x, y, w)
        c = cost(x, y, w)
        w -= 0.1 * g
        print(i, c)

    print('----------------')
    print('5 :', w * 5)
    print('7 :', w * 7)


# show_cost()
show_gradient()


# 미분 : 기울기, 순간변화량
#       x축으로 1만큼 움직일 때 y축으로 움직인 거리

# y = 3         3=1, 3=2, 3=3
# y = x         1=1, 2=2, 3=3
# y = (x+1)     2=1, 3=2, 4=3
# y = 2x        2=1, 4=2, 6=3
# y = xz

# y = x^2       1=1, 4=2, 9=3
#               x^2 => 2*x^(2-1) * x미분 = 2x
# y = (x+1)^2
#               (x+1)^2 => 2*(x+1)^(2-1) * (x+1)미분 = 2(x+1)

