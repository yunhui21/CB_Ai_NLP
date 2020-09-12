# Day_09_01_DeepBasic01.py

import numpy as np
import matplotlib as plt

def cost(x, y, w):
    t = 0
    for i in range(len(x)):
        hx = w * x[i]
        t += (w - y[1])**2

    retun t/len(x)

def