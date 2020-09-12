# Day_06_05_matplotlib.py
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np


def plot_1():
    # plt.plot([1, 2, 3, 4])

    x = [1, 2, 3, 4]
    # plt.plot(x, x)
    # plt.plot(x, x, 'r')   # line
    plt.plot(x, x, 'o')     # scatter
    plt.show()


# 문제
# y = x^2 그래프를 그려보세요
def plot_2():
    x = np.arange(-5, 5, 0.1)
    plt.plot(x, x ** 2)
    plt.plot(x, x ** 2, 'rx')
    plt.show()


# 문제
# 로그 그래프 4개를 그려보세요
def plot_3():
    x = np.arange(0.1, 3, 0.1)
    plt.plot(x, np.log(x))
    plt.plot(x, -np.log(x))

    plt.plot(-x, np.log(x))
    plt.plot(-x, -np.log(x))
    plt.show()


def plot_4():
    x = np.arange(0.1, 3, 0.1)
    plt.subplot(1, 2, 1)
    plt.plot(x, np.log(x))
    plt.subplot(1, 2, 2)
    plt.plot(x, -np.log(x))

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(-x, np.log(x))
    plt.subplot(2, 2, 4)
    plt.plot(-x, -np.log(x))
    plt.show()


# 문제
# 여성 데이터를 표시하세요
def plot_5():
    men = [23, 31, 27, 19, 26]
    women = [32, 37, 20, 25, 33]

    indices = np.arange(len(men))
    # temp = [0.5, 1.5, 2.5, 3.5, 4.5]

    # plt.bar(indices, men, width=0.45, color='rgb')
    # plt.bar(indices, men, width=0.45, color=['red', 'green'])
    # plt.bar(indices, men, width=0.45, color=colors.BASE_COLORS)
    plt.bar(indices, men, width=0.45, color=colors.TABLEAU_COLORS)
    plt.bar(indices+0.5, women, width=0.45)
    # plt.bar(temp, women, width=0.45)
    plt.show()


# plot_1()
# plot_2()
# plot_3()
# plot_4()
plot_5()

