# Day_18_01_17flowers.py
# 문제
# origin 폴더에 있는 이미지를 224호 축소해서 224 폴더로 복사하세요.
# 이름을 17flowers_origin
import tensorflow as tf
import numpy as np
from PIL import Image # pillow
# import PIL 자동완성기능이 안된다.
import os

# 문제
# 꽃 폴더로부터 파일을 읽어서 x, y 데이터를 만드세요.
# y: 80개 간격이 나누어져 있습니다. (0-16)
# x: 4차원 배열

def make_flowers_224(src_folder, dst_folder, img_size=[224, 224]):
    for filename in os.listdir(src_folder):
        print(filename)

        items = filename.split('.')
        if items[-1] != 'jpg':
            continue

        # file_path = src_folder + '/' + filename
        # print(file_path)
        file_path = os.path.join(src_folder, filename)
        print(file_path)
        image1 = Image.open(file_path)
        image2 = image1.resize(img_size)
        image2.save(os.path.join(dst_folder, filename))


def make_flowers_xy(dir_name):
    x , y = [], []
    for filename in os.listdir(dir_name):
        items = filename.split('.')
        if items[-1] != 'jpg':
            continue

        idx = int(items[0].split('_')[1])
        print(idx, (idx-1) // 80)
        y.append((idx - 1)// 80)

        file_path = os.path.join(dir_name, filename)
        img = Image.open(file_path)
        # print(type(img))
        # print(type(np.array(img)), np.array(img).shape)

    return np.array(x), np.array(y), len(set(y))

# make_flowers_224('data/17flowers_origin', 'data/17flowers_224', img_size=[56, 56])
x, y, n_classes = make_flowers_xy('17flowers_224')

