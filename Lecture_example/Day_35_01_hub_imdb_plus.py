# Day_35_01_hub_imdb_plus.py
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

# 문제
# 케라스에서 제공하는 imdb 데이터셋을 텐서플로 허브와 연동해서 결과를 구현하세요


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)
print(x_train.shape, x_test.shape)      # (25000,) (25000,)
print(y_train.shape, y_test.shape)      # (25000,) (25000,)

print(x_train[:3])
# [list([1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, ...])
#  list([1, 194, 1153, 194, 8255, 78, 228, 5, 6, 1463, 4369, ...])
#  list([1, 14, 47, 8, 30, 31, 7, 4, 249, 108, 7, 4, 5974, ...])]

print(y_train[:10])     # [1 0 0 1 0 0 1 0 1 0]

word_index = tf.keras.datasets.imdb.get_word_index()
index_word = {v: k for k, v in word_index.items()}
# print(word_index)     # {'wayan': 30528, 'ballads': 14348, ...}
# print(index_word)     # {30528: 'wayan', 14348: 'ballads', ...}

# train_data, test_data = [], []
# for item in x_train:
#     # print(item)                             # [1, 14, 22, 16, ...]
#     # print([index_word[i] for i in item])    # ['the', 'as', 'you', 'with', ...]
#     train_data.append(' '.join([index_word[i] for i in item]))
#
# for item in x_test:
#     test_data.append(' '.join([index_word[i] for i in item]))
#
# print(train_data[-1])   # the movie is thought completely br of ...

x_train = [' '.join([index_word[i] for i in item]) for item in x_train]
x_test  = [' '.join([index_word[i] for i in item]) for item in x_test ]

# ------------------------- #
# 아래는 이전 파일에서 복사한 코드

url = 'https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1'
hub_layer = hub.KerasLayer(url,
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

# validation_split 옵션은 텐서와 넘파이에 대해서만 지원
x_train = np.array(x_train)
x_test = np.array(x_test)

model.fit(x_train, y_train,
          epochs=20,
          validation_split=0.4,
          verbose=2)
print(model.evaluate(x_test, y_test, verbose=0))
