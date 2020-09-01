# Day_21_04_GradientBias.py
# Day_16_01_dl_basic.py
import matplotlib.pyplot as plt


def cost(x, y, w, b):
    c = 0
    for i in range(len(x)):
        hx = w * x[i] + b
        loss = (hx - y[i]) ** 2     # mse
        c += loss

    return c / len(x)


def gradient_descent(x, y, w, b):
    grad0, grad1 = 0, 0
    for i in range(len(x)):
        hx = w * x[i] + b
        grad0 += (hx - y[i]) * x[i]
        grad1 += (hx - y[i])

    return grad0 / len(x), grad1 / len(x)


def show_gradient_bias():
    x = [1, 2, 3]
    y = [1, 2, 3]

    w, b = 5, -3
    for i in range(1000):
        c = cost(x, y, w=w, b=b)
        g0, g1 = gradient_descent(x, y, w=w, b=b)
        w -= 0.1 * g0
        b -= 0.1 * g1
        print(i, c)

    # 문제
    # x가 5와 7일 때의 결과를 알려주세요
    print('5 :', w * 5 + b)
    print('7 :', w * 7 + b)


show_gradient_bias()
