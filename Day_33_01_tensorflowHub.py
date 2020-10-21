# Day_33_01_tensorflowHub.py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import tensorflow_hub as hub


def get_image_classfier():

    url = 'https://tfhub.dev/tensorflow/resnet_50/classification/1'

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input([224, 224, 3]))
    model.add(hub.KerasLayer(url)) # 전이학습과 같은 방식
    # model.add(hub.keras_layer(url))       # 에러

    labels_path = tf.keras.utils.get_file(
        'ImageNetLabels.txt',
        'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
    )

    labels = np.array(open(labels_path).read().splitlines())

    return model, labels


def get_resnet50():

    url = 'https://tfhub.dev/tensorflow/resnet_50/feature_vector/1'

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input([224, 224, 3]))
    model.add(hub.KerasLayer(url, trainable=False)) # 전이학습과 같은 방식
    model.add(tf.keras.layers.Dense(5, activation='softmax'))

    return model


# 문제
# resnet50모델의 피쳐백터를 사용해서 꼿 데이터 첫번째 배치를 구하세요.
'''
'''

def simple_transfer_learning():
    images_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
    images_path = tf.keras.utils.get_file('flower_photos', images_url, untar=True)
    print(images_path) # C:\Users\USER\.keras\datasets\flower_photos

    test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)

    test_flow = test_gen.flow_from_directory(images_path,
                                             target_size=[224, 224],
                                             batch_size=32,
                                             class_mode='sparse')
    x, y = test_flow.next()
    print(x.shape, y.shape)
    model = get_resnet50()
    preds = model.predict(x, verbose=0)  # 4차원데이터를 가져야 한다.
    print(preds.shape)

    print(test_flow.class_indices) # {'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}

    #
    # preds_arg = np.argmax(preds, axis=1)
    #
    # plt.figure(figsize=(10, 8))
    # for i, (img, label, pred) in enumerate(zip(x, y, preds_arg)):
    #     # print(img.shape, label, pred)
    #
    #     plt.subplot(4, 8, i+1)
    #     plt.title(labels[pred])
    #     plt.axis('off')
    #     plt.imshow(img)
    # plt.show()


simple_transfer_learning()