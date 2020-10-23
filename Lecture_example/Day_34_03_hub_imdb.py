# Day_34_03_hub_imdb.py
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

# 문제
# 텐서플로 허브에 있는 swivel 임베딩을 사용해서
# 텐서플로 데이터셋에서 제공하는 imdb 리뷰에 대해 정확도를 예측하세요

# 문제
# gnews-swivel-20dim 외에 추가적인 임베딩을 사용해서 결과를 비교하세요
# (gnews-swivel-20dim-with-oov, nnlm-en-dim50, nnlm-en-dim128)


def embed_basic():
    url = 'https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1'
    sents = ['cat is on the mat', 'dog is in the fog']

    embeddings = hub.load(url)
    hub_layer = hub.KerasLayer(url)

    print(embeddings(sents))
    print(embeddings(sents).shape)      # (2, 20)
    print(hub_layer(sents))


# 처음 버전
def show_imdb_by_swivel():
    train_set, valid_set, test_set = tfds.load(
        'imdb_reviews', as_supervised=True,
        split=['train[:60%]', 'train[60%:]', 'test'])

    url = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
    hub_layer = hub.KerasLayer(url,
                               output_shape=[20],
                               input_shape=[], dtype=tf.string)

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

    model.fit(train_set,
              epochs=20,
              validation_data=valid_set)
    print(model.evaluate(test_set, verbose=0))


def show_imdb(train_set, valid_set, test_set, url):
    # trainable을 False로 주면, 뒤쪽 모델로 갈수록 성능 차이가 많이 난다
    # (순서대로 60%, 69%, 74%, 78%)
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

    model.fit(train_set,
              epochs=20,
              validation_data=valid_set,
              verbose=2)
    print(model.evaluate(test_set, verbose=0))


train_set, valid_set, test_set = tfds.load(
    'imdb_reviews', as_supervised=True,
    split=['train[:60%]', 'train[60%:]', 'test'])

# embed_basic()
# show_imdb_by_swivel()

show_imdb(train_set, valid_set, test_set, 'https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1')
show_imdb(train_set, valid_set, test_set, 'https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim-with-oov/1')
show_imdb(train_set, valid_set, test_set, 'https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1')
show_imdb(train_set, valid_set, test_set, 'https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1')

# trainable을 True로 설정 (모델별로 차이가 심하지 않다)
# 아래 2개는 학습이 부족한 것으로 판단할 수 있다. 더 많은 에포크 필요.
# [0.339221328496933, 0.8544800281524658]
# [0.3258782923221588, 0.8621199727058411]
# [0.47940558195114136, 0.8518000245094299]
# [0.5330076813697815, 0.8507999777793884]

