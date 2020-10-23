# Day_35_01_hubimdb_plus.py
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn import model_selection
# 문제
# 케라스에서 제공하는 imdb hub와 연동해서 결과를 구현하세요.

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)

word_index = tf.keras.datasets.imdb.get_word_index()
print(word_index)


