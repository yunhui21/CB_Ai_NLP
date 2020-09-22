# Day_21_01_CarEvaluation.py

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection

def get_car_dense():

    cars = pd.read_csv('../data/car.data', 'r', header=None, names=['buying','maint','doors','persons','lug_boot','safety', 'eval'])
    print(cars)
    lb = preprocessing.label_binarize()
    buying = lb.fit_transform(cars.buying)
    maint  = lb.fit_transform(cars.maint)
    doors  = lb.fit_transform(cars.doors)




get_car_dense()