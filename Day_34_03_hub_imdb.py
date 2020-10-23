# Day_34_03_hub_imdb.py

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

# tensorflow 허브에 있는 swivel 임베딩을 사용해서 tfds에서 제공하는 imdb_reviews의 정확도를 예측하세요.


# 문제
# 사용해서 결과를 예측하세요.
# swivel:
#
# nnlm : tf2-preview/nnlm-es-dim50-with-normalization

def embed_basic():
    #swivel
    url ='https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim-with-oov/1'
    sents = ['cat is set the mat', 'day is in the fog']

    # first
    embedding = hub.load(url)
    # second
    hub_layer = hub.KerasLayer(url)

    print(embedding(sents))
    print(embedding(sents).shape) # (2, 20)

    print(hub_layer(sents))

def show_imdb_by_swivel():

    train_set, valid_set, test_set = tfds.load('imdb_reviews',
                                               as_supervised=True,
                                               split=['train[:60%]', 'train[60%:]', 'test'])  # Dataset에서만 사용 가능한 문법
    url = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim-with-oov/1"
    hub_layer = hub.KerasLayer(url, output_shape=[20],
                               input_shape=[], dtype=tf.string)

    model = tf.keras.Sequential()
    model.add(hub_layer)
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer = 'adam',
                  loss = tf.keras.losses.binary_crossentropy,
                  metrics =['acc'])
    train_set = train_set.shuffle(10000).batch(512)
    valid_set = valid_set.batch(512)
    test_set = test_set.batch(512)
    model.fit(train_set, epochs=20,validation_data=valid_set)
    model.evaluate(test_set, verbose=0)

train_set, valid_set, test_set = tfds.load('imdb_reviews',
                                               as_supervised=True,
                                               split=['train[:60%]', 'train[60%:]', 'test'])  # Dataset에서만 사용 가능한 문법


def show_imdb(train_set, valid_set, test_set, url):
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
    train_set = train_set.shuffle(10000).batch(512)
    valid_set = valid_set.batch(512)
    test_set = test_set.batch(512)
    model.fit(train_set, epochs=20, validation_data=valid_set, verbose=0)
    model.evaluate(test_set, verbose=0)


# show_imdb_by_swivel()
show_imdb(train_set, valid_set, test_set, 'https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1')
show_imdb(train_set, valid_set, test_set, 'https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim-with-oov/1')
show_imdb(train_set, valid_set, test_set, 'https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1')
show_imdb(train_set, valid_set, test_set, 'https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1')
