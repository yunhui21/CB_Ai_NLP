# Day_35_01_hubimdb_plus.py
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn import model_selection
import tensorflow_hub as hub
import numpy as np
# 문제
# 케라스에서 제공하는 imdb hub와 연동해서 결과를 구현하세요.

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)

# print(x_train.shape, x_test.shape)
# print(y_train.shape, y_test.shape)
#
# print(x_train[:3])
# # [list([1, 14, 22, 16, 43, 530, 973,...]) 원래상태로 돌려놓도록 해야한다.
# # list([1, 194, 1153, 194, 8255, 78, 228...])
# print(y_train[:10]) # [1 0 0 1 0 0 1 0 1 0]

word_index = tf.keras.datasets.imdb.get_word_index()
index_word = {v: k for k, v in word_index.items()}

# print(word_index)     # {'wayan': 30528, 'ballads': 14348, ...}
# print(index_word)     # {30528: 'wayan', 14348: 'ballads', ...}
# train_data, test_data = [], []
# for item in x_train:
#     # print(item)
#     # print([index_word[i] for i in item])
#     train_data.append(' '.join([index_word[i] for i in item]))
#
# print(train_data[-1])
#
# for item in x_test:
#     # print(item)
#     # print([index_word[i] for i in item])
#     test_data.append(' '.join([index_word[i] for i in item]))
#
# print(test_data[-1])

x_train = [' '.join([index_word[i] for i in item]) for item in x_train]
x_test = [' '.join([index_word[i] for i in item]) for item in x_test]


#-------------------------------------------------------------------------#
# 이전코드에서 복사한 코드


url = 'https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1'
hub_layer = hub.KerasLayer(url,
                               # output_shape=[20],
                               input_shape=[],
                               dtype=tf.string,
                               trainable=True)

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss=tf.keras.losses.binary_crossentropy,
              metrics=['acc'])
x_train = np.array(x_train)
x_test  = np.array(x_test)
model.fit(x_train, y_train,
          epochs=20,
          validation_split=0.4,
          verbose=2)
print(model.evaluate(x_test, y_test, verbose=0))
# [0.7376493215560913, 0.8451200127601624]
