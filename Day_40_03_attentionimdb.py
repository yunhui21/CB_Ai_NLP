# Day_40_03_attentionimdb.py

import tensorflow as tf
import numpy as np

num_words, max_len, embed_size = 5000, 500, 32

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words = num_words)
print(x_train.shape, x_test.shape) # (25000,) (25000,)
print(y_train.shape, y_test.shape) # (25000,) (25000,)

print(x_train[:3]) # [list([1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, 66, 3941, 4, 173, 36,...])]

print(y_train[:10]) # [1 0 0 1 0 0 1 0 1 0]