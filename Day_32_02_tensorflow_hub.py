# Day_32_02_tensorflow_hub.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import tensorflow_hub as hub


# 문제
# get_hub classifier 함수에서 만든 모델을 사용해서
#
# tensorflow-hub : 허브의 역할
def get_image_classfier():

    # url = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4'
    # url = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2'
    url = 'https://tfhub.dev/tensorflow/resnet_50/classification/1'

# 문제
# resnet 5.0을 가져오세요.

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input([224, 224, 3]))
    model.add(hub.KerasLayer(url)) # 전이학습과 같은 방식
    # model.add(hub.keras_layer(url))       # 에러

    # Day_23_02_WordRnnChosun.py 파일에서 사용
    labels_path = tf.keras.utils.get_file(
        'ImageNetLabels.txt',
        'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
    )

    # print(labels_path) # C:\Users\USER\.keras\datasets\ImageNetLabels.txt

    labels = np.array(open(labels_path).read().splitlines())
    # print(labels) # ['background' 'tench' 'goldfish' ... 'bolete' 'ear' 'toilet tissue']
    # print(len(labels)) # 1001
    # labels = np.array(open(labels_path).readlines())
    # print(labels)
    return model, labels


def classify_Image():
    img_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg'
    img_path = tf.keras.utils.get_file('grace_hopper.jpg', img_url)

    img_hopper = Image.open(img_path).resize([224, 224])
    print(img_hopper)  # <PIL.Image.Image image mode=RGB size=224x224 at 0x215A5980388>


    # plt.imshow(img_hopper)
    # plt.show()

    array_hopper = np.array(img_hopper) # 0 ~ 1 사이로 만들어졌다.
    print(array_hopper.shape) # (224, 224, 3)

    plt.subplot(1, 2, 1)
    plt.title('original')
    plt.imshow(array_hopper)
    # plt.imshow(np.float32(array_hopper)) # int 0~255, float32 0~1 : 0 dark, 1 bright


    print(np.min(array_hopper), np.max(array_hopper)) # 0 255
    # array_scaled = array_hopper / 255
    # array_scaled = array_hopper / 510 # dark
    array_scaled = array_hopper/127     # bright

    model, labels = get_image_classfier()
    preds = model.predict(array_scaled[np.newaxis]) # 4차원데이터를 가져야 한다.
    print(preds.shape) # (1, 1001)

    # pos = np.argmax(preds, axis=1)
    # pos = np.argmax(preds, axis=-1)

    pos = np.argmax(preds)
    # scaled = 653, non_scaled = 722
    print(pos)
    print('predicted:', labels[pos])

    plt.subplot(1, 2, 2)
    plt.title(labels[pos])
    plt.imshow(array_scaled)
    plt.show()


def classify_by_generator():
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
    model, labels = get_image_classfier()
    preds = model.predict(x, verbose=0)  # 4차원데이터를 가져야 한다.
    print(preds.shape)

    preds_arg = np.argmax(preds, axis=1)

    plt.figure(figsize=(10, 8))
    for i, (img, label, pred) in enumerate(zip(x, y, preds_arg)):
        # print(img.shape, label, pred)

        plt.subplot(4, 8, i+1)
        plt.title(labels[pred])
        plt.axis('off')
        plt.imshow(img)
    plt.show()



classify_Image()
# classify_by_generator()