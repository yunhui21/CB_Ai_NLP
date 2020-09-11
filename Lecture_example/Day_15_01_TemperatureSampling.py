# Day_15_01_TemperatureSampling.py
import numpy as np

np.set_printoptions(precision=2, floatmode='fixed', suppress=True)

def softmax_1(dist):
    return dist / np.sum(dist)


def softmax_2(dist):
    dist = np.exp(dist)
    return dist / np.sum(dist)


def temperature_pick(dist, temperature):
    dist = np.log(dist) / temperature
    dist = np.exp(dist)
    return dist / np.sum(dist)


dist = [2.0, 1.0, 0.1]

print(softmax_1(dist))
print(softmax_2(dist))

for t in np.linspace(0.1, 1.0, 10):
    print('{:.1f}'.format(t), temperature_pick(dist, t))


