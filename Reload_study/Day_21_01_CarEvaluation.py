# Day_21_01_CarEvaluation.py

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection

def get_car_dense():

    cars = pd.read_csv('data/carse.data', 'r', header=None)
    print(cars)
    lb = preprocessing.label_binarize()




get_car_dense()