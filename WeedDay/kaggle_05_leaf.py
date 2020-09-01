# Kaggle_05_leaf.py
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import preprocessing, model_selection, decomposition
import os
import matplotlib.pyplot as plt
from PIL import Image
import PIL
import cv2


# 문제 1
# 타이타닉의 코드를 나뭇잎에 적용하세요 (소프트맥스)
# 두 번째 리더보드 버전을 구현하세요
# kaggle_02_titanic.py

# 문제 2
# 서브미션 파일 생성 함수를 구현하세요

# 문제 3
# 이미지 폴더로부터 사진을 읽어서 피처를 생성하는 함수를 만드세요
# 피처 : 너비, 높이, 면적, 비율, 수직/수평

# 문제 4
# 새로 만든 피처(pca)를 기존 피처에 연결하세요

# 문제 5
# 새로 만든 피처(moments)를 기존 피처에 연결하세요


def make_basic_features():
    dir_path = 'kaggle/leaf_images'
    features = {}

    for filename in os.listdir(dir_path):
        # file_path = os.path.join(dir_path, filename)
        file_path = dir_path + '/' + filename
        # print(file_path)        # kaggle/leaf_images\999.jpg

        lead_id = int(filename.split('.')[0])
        # print(lead_id)

        img = plt.imread(file_path)
        # print(img)
        # print(type(img))    # <class 'numpy.ndarray'>
        # print(img.shape)    # (467, 526)

        w, h = img.shape
        area = w * h
        ratio = h / w
        portrait = int(h > w)

        features[lead_id] = (w, h, area, ratio, portrait)

    return features


def make_pca_features():
    dir_path = 'kaggle/leaf_images'

    leaf_ids, images = [], []
    for filename in os.listdir(dir_path):
        file_path = dir_path + '/' + filename

        leaf_id = int(filename.split('.')[0])

        img = Image.open(file_path)
        img = img.resize((50, 50), PIL.Image.ANTIALIAS)
        img = np.uint8(img).reshape(-1)

        leaf_ids.append(leaf_id)
        images.append(img)

    pca = decomposition.PCA(n_components=35)
    pca_features = pca.fit_transform(images)
    # print(pca_features.shape)       # (1584, 35)
    # print(pca_features.dtype)       # (1584, 35)
    # print(len(leaf_ids))

    features = {}
    for leaf_id, feat in zip(leaf_ids, pca_features):
        features[leaf_id] = tuple(feat)

    return features


def make_moments_features():
    dir_path = 'kaggle/leaf_images'
    features = {}

    for filename in os.listdir(dir_path):
        file_path = dir_path + '/' + filename
        lead_id = int(filename.split('.')[0])

        img = cv2.imread(file_path, 0)
        ret, thresh = cv2.threshold(img, 127, 255, 0)
        contours, hierachy = cv2.findContours(thresh, 1, 2)
        cnt = contours[0]
        moments =cv2.moments(cnt)

        # moments : dict
        features[lead_id] = tuple(moments.values())

    return features


def append_new_features(ids, origin, new_features):
    # print(origin.shape, len(new_features))

    # 아래 반복문을 컴프리헨션으로 바꾸세요
    # items = []
    # for leaf_id in ids:
    #     values = new_features[leaf_id]
    #     items.append(values)

    items = [new_features[leaf_id] for leaf_id in ids]
    # print(len(items))

    origin = np.hstack([origin, items])
    # print(origin.shape, origin.dtype)

    return origin


def get_data_train():
    leaf = pd.read_csv('kaggle/leaf_train.csv', index_col=0)
    # print(leaf)
    # leaf.info()

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(leaf.species)

    leaf.drop(['species'], axis=1, inplace=True)
    x = leaf.values

    return x, y, le.classes_, leaf.index.values


def get_data_test():
    leaf = pd.read_csv('kaggle/leaf_test.csv', index_col=0)

    x = leaf.values
    return x, leaf.index.values


def make_submission(file_path, ids, preds, leaf_names):
    f = open(file_path, 'w', encoding='utf-8')

    print('id', *leaf_names, sep=',', file=f)
    for leaf_id, pred in zip(ids, preds):
        print(leaf_id, *pred, sep=',', file=f)

    f.close()


def multi_layers(ph_x, ph_d, layers, input_size):
    d = ph_x
    n_features = input_size
    for n_classes in layers:
        w = tf.get_variable(str(np.random.rand()),
                            shape=[n_features, n_classes],
                            initializer=tf.glorot_uniform_initializer)
        b = tf.Variable(tf.zeros([n_classes]))

        z = tf.matmul(d, w) + b

        if n_classes == layers[-1]:     # 1
            break

        r = tf.nn.relu(z)
        d = tf.nn.dropout(r, keep_prob=ph_d)

        n_features = n_classes

    return z, tf.nn.softmax(z)


def model_leaf(x_train, y_train, x_valid, x_test):
    ph_x = tf.placeholder(tf.float32)
    ph_y = tf.placeholder(tf.int32)
    ph_d = tf.placeholder(tf.float32)   # drop-out

    z, hx = multi_layers(ph_x, ph_d,
                         layers=[160, 128, 99],
                         input_size=x_train.shape[1])

    loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=ph_y, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.AdamOptimizer(0.001)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    epochs = 200
    batch_size = 32
    n_iteration = len(x_train) // batch_size

    indices = np.arange(len(x_train))

    for i in range(epochs):
        np.random.shuffle(indices)

        c = 0
        for j in range(n_iteration):
            n1 = j * batch_size
            n2 = n1 + batch_size

            part = indices[n1:n2]

            xx = x_train[part]
            yy = y_train[part]

            sess.run(train, {ph_x: xx, ph_y: yy, ph_d: 0.7})
            c += sess.run(loss, {ph_x: xx, ph_y: yy, ph_d: 0.7})

        # if i % 10 == 0:
        print(i, c / n_iteration)

    preds_valid = sess.run(hx, {ph_x: x_valid, ph_d: 1.0})
    preds_test = sess.run(hx, {ph_x: x_test, ph_d: 1.0})
    sess.close()

    return preds_valid, preds_test


def show_accuracy_sparse(preds, labels):
    preds_arg = np.argmax(preds, axis=1)
    # y_arg = np.argmax(labels, axis=1)

    equals = (preds_arg == labels)
    print('acc :', np.mean(equals))


x, y, leaf_names, ids_train = get_data_train()
x_test, ids_test = get_data_test()

basic_features = make_basic_features()
pca_features = make_pca_features()
moments_features = make_moments_features()

x = append_new_features(ids_train, x, basic_features)
x_test = append_new_features(ids_test, x_test, basic_features)

x = append_new_features(ids_train, x, pca_features)
x_test = append_new_features(ids_test, x_test, pca_features)

x = append_new_features(ids_train, x, moments_features)
x_test = append_new_features(ids_test, x_test, moments_features)

# scaler = preprocessing.MinMaxScaler()
scaler = preprocessing.StandardScaler()
scaler.fit(x)

x = scaler.transform(x)
x_test = scaler.transform(x_test)

data = model_selection.train_test_split(x, y, train_size=0.7)
x_train, x_valid, y_train, y_valid = data

# print(x_train.shape, x_valid.shape, x_test.shape)  # (693, 192) (297, 192) (594, 192)
# print(y_train.shape, y_valid.shape)                # (693,) (297,)
# print(len(leaf_names))                             # 99

model_leaf(x_train, y_train, x_valid, x_test)
preds_valid, preds_test = model_leaf(x_train, y_train, x_valid, x_test)
show_accuracy_sparse(preds_valid, y_valid)

make_submission('kaggle/leaf_submission_1.csv',
                ids_test, preds_test, leaf_names)

# preds_max = np.max(preds_test, axis=1)
# print(preds_max)
# print(preds_max.shape)
#
# sorted_max = np.sort(preds_max)
# print(sorted_max)
# print(sorted_max[:10])
# print(sorted_max[-10:])

eye = np.eye(99, dtype=np.float32)
for i in range(len(preds_test)):
    p = preds_test[i]
    m = np.max(p)
    if m > 0.98:
        n = np.argmax(p)
        preds_test[i] = eye[n]

make_submission('kaggle/leaf_submission_2.csv',
                ids_test, preds_test, leaf_names)

