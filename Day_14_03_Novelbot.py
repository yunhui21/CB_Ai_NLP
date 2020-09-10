# Day_14_03_Novelbot.py

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing, model_selection
# 문제 1
# 니체의 저서를 읽고 니체풍으로 새로운 소설을 쓰세요.
# BasicRnn_8과 stock 코드를 병합해서 사용하도록 합니다.

# 문제 2
# 결과를

f = open('data/nietzsche.txt', 'r', encoding='utf-8')
long_text = f.read(1000).lower()
f.close()

print(long_text)

lb = preprocessing.LabelBinarizer()
onehot = lb.fit_transform(list(long_text))

seq_len = 60
rng = [(i, i+seq_len) for i in range(len(long_text) - seq_len)]

x = [onehot[s:e] for s, e in rng]
y = [onehot[e] for s, e in rng]
print(y[:3])

x = np.float32(x)
y = np.argmax(y, axis=1)
print(x.shape, y.shape)

model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dense(len(lb.classes_), activation=tf.keras.activations.softmax)
    ])

model.compile(optimizer=tf.keras.optimizers.Adam(0.1),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              matrics=['acc'])
model.fit(x, y, epochs=10, verbose=2, batch_size=128)

preds = model.predict(x)

preds_arg = np.argmax(preds, axis=1)
print(preds_arg.shape)

print(long_text)
print(''.join(lb.classes_[preds_arg]))
#
# for p in preds_arg:
#     print(lb.classes_[p], end='')