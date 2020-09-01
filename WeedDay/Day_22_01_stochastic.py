# Day_22_01_stochastic.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 문제
# action.txt 파일로부터 x, y를 반환하는 함수를 만드세요
def get_xy():
    action = pd.read_csv('data/action.txt',
                         header=None,
                         names=['bias', 'feature1', 'feature2', 'label'])
    print(action)

    y = action.label.values
    print(y[:3])

    action.drop(['label'], axis=1, inplace=True)
    x = action.values

    return x, y.reshape(-1, 1)


def get_xy_np():
    # action = np.loadtxt('data/action.txt',
    #                     delimiter=',',
    #                     unpack=True)
    # print(action.shape)

    action = np.loadtxt('data/action.txt',
                        delimiter=',')
    print(action.shape)

    return action[:, :-1], action[:, -1:]


def gradient_descent(x, y):
    m, n = x.shape              # 100, 3
    w = np.zeros([n, 1])        # (3, 1)
    lr = 0.01

    for i in range(m):
        z = np.dot(x, w)        # (100, 1) = (100, 3) @ (3, 1)
        h = sigmoid(z)          # (100, 1)
        e = h - y               # (100, 1) = (100, 1) - (100, 1)
        g = np.dot(x.T, e)      # (3, 1) = (3, 100) @ (100, 1)
        w -= lr * g             # (3, 1) -= scalar * (3, 1)

    return w.reshape(-1)        # (3,)


def stochastic(x, y):
    m, n = x.shape              # 100, 3
    w = np.zeros(n)             # (3,)
    lr = 0.01

    for i in range(m * 10):
        p = i % m
        z = np.sum(x[p] * w)    # scalar = sum((3,) * (3,))
        h = sigmoid(z)          # scalar
        e = h - y[p]            # scalar = scalar - scalar
        g = x[p] * e            # (3,) = (3,) * scalar
        w -= lr * g             # (3,) -= scalar * (3,)

    return w                    # (3,)


def stochastic_random(x, y):
    m, n = x.shape              # 100, 3
    w = np.zeros(n)             # (3,)
    lr = 0.01

    for i in range(m * 10):
        p = np.random.randint(0, m)
        z = np.sum(x[p] * w)    # scalar = sum((3,) * (3,))
        h = sigmoid(z)          # scalar
        e = h - y[p]            # scalar = scalar - scalar
        g = x[p] * e            # (3,) = (3,) * scalar
        w -= lr * g             # (3,) -= scalar * (3,)

    return w                    # (3,)


def mini_batch(x, y):
    m, n = x.shape              # 100, 3
    w = np.zeros([n, 1])        # (3, 1)
    lr = 0.01
    epochs = 10
    batch_size = 5

    for i in range(epochs):
        n_iteration = m // batch_size
        for j in range(n_iteration):
            n1 = j * batch_size
            n2 = n1 + batch_size

            z = np.dot(x[n1:n2], w)     # (5, 1) = (5, 3) @ (3, 1)
            h = sigmoid(z)              # (5, 1)
            e = h - y[n1:n2]            # (5, 1) = (5, 1) - (5, 1)
            g = np.dot(x[n1:n2].T, e)   # (3, 1) = (3, 5) @ (5, 1)
            w -= lr * g                 # (3, 1) -= scalar * (3, 1)

    return w.reshape(-1)                # (3,)


def mini_batch_random(x, y):
    m, n = x.shape              # 100, 3
    w = np.zeros([n, 1])        # (3, 1)
    lr = 0.01
    epochs = 10
    batch_size = 5

    for i in range(epochs):
        n_iteration = m // batch_size
        for j in range(n_iteration):
            n1 = j * batch_size
            n2 = n1 + batch_size

            z = np.dot(x[n1:n2], w)     # (5, 1) = (5, 3) @ (3, 1)
            h = sigmoid(z)              # (5, 1)
            e = h - y[n1:n2]            # (5, 1) = (5, 1) - (5, 1)
            g = np.dot(x[n1:n2].T, e)   # (3, 1) = (3, 5) @ (5, 1)
            w -= lr * g                 # (3, 1) -= scalar * (3, 1)

        # np.random.shuffle(x)
        # np.random.shuffle(y)

        indices = np.arange(m)
        np.random.shuffle(indices)

        x = x[indices]
        y = y[indices]

    return w.reshape(-1)                # (3,)


def decision_boundary(w, c, label):
    b, w1, w2 = w
    y1 = -(w1 * -4 + b) / w2
    y2 = -(w1 *  4 + b) / w2

    plt.plot([-4, 4], [y1, y2], c, label=label)

    # y = w1 * x1 + w2 * x2 + b
    # 0 = w1 * x1 + w2 * x2 + b
    # 0 = w1 * x + w2 * y + b
    # w2 * y = -(w1 * x + b)
    # y = -(w1 * x + b) / w2


# x, y = get_xy()
x, y = get_xy_np()
# print(x.shape, y.shape)     # (100, 3) (100, 1)

# w = gradient_descent(x, y)
# print(w)

# 문제
# x, y 데이터를 그래프로 표현하세요
# for i in range(len(x)):
#     if y[i][0] == 0:
#         plt.plot(x[i][1], x[i][2], 'ro')
#     else:
#         plt.plot(x[i][1], x[i][2], 'gx')

for i in range(len(x)):
    _, x1, x2 = x[i]
    plt.plot(x1, x2, 'ro' if y[i][0] else 'gx')
    # if y[i][0] == 0:
    #     plt.plot(x1, x2, 'ro')
    # else:
    #     plt.plot(x1, x2, 'gx')

decision_boundary(gradient_descent(x, y), 'r', 'base')
decision_boundary(stochastic(x, y), 'g', 'stoc')
decision_boundary(stochastic_random(x, y), 'b', 'stoc random')
decision_boundary(mini_batch(x, y), 'k', 'mini-batch')
decision_boundary(mini_batch_random(x, y), 'm', 'mini-batch-random')

plt.legend()
plt.show()




