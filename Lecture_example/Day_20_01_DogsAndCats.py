# Day_20_01_DogsAndCats.py
import tensorflow as tf
import os
import shutil

# print('a' * 3)
# print(['a'] * 3)
# # print({'a'} * 3)          # error
# # print({'a': 12} * 3)      # error
# print([{'a': 12}] * 3)
#
# [{
#       'depth': base_depth * 4,
#       'depth_bottleneck': base_depth,
#       'stride': 1
#   }] * (num_units - 1) + [{
#       'depth': base_depth * 4,
#       'depth_bottleneck': base_depth,
#       'stride': stride
#   }]

# 폴더 구조
# dogs-vs-cats
#     small
#         test
#             cats
#             dogs
#         train
#             cats
#             dogs
#         valid
#             cats
#             dogs
#     train

# 문제 1
# 위와 같은 폴더 구조를 만드세요
# 폴더가 이미 만들어져 있다면 만들지 않습니다

# 문제 2
# train 폴더에 1000개, valid에 500개, test에 500개씩 옮기세요
# (파일 이름에 들어있는 규칙을 활용하세요)


def make_folder_structure():
    def make_folder_if_not_exist(dir_path):
        if not os.path.exists(dir_path):
            # os.mkdir(dir_path)    # 반드시 상위 폴더가 존재해야 함
            os.makedirs(dir_path)   # 상위 폴더가 없어도 하위 폴더까지 생성

    # make_folder_if_not_exist('dogs-vs-cats/small')

    # make_folder_if_not_exist('dogs-vs-cats/small/test')
    # make_folder_if_not_exist('dogs-vs-cats/small/train')
    # make_folder_if_not_exist('dogs-vs-cats/small/valid')

    make_folder_if_not_exist('dogs-vs-cats/small/test/cats')
    make_folder_if_not_exist('dogs-vs-cats/small/test/dogs')
    make_folder_if_not_exist('dogs-vs-cats/small/train/cats')
    make_folder_if_not_exist('dogs-vs-cats/small/train/dogs')
    make_folder_if_not_exist('dogs-vs-cats/small/valid/cats')
    make_folder_if_not_exist('dogs-vs-cats/small/valid/dogs')


def make_small_datset():
    def copy_data(animal, start, end, target):
        dst_folder = os.path.join('dogs-vs-cats/small/', target)
        for i in range(start, end):
            filename = '{}.{}.jpg'.format(animal, i)
            # print(filename)

            src_path = os.path.join('dogs-vs-cats/train/', filename)
            dst_path = os.path.join(dst_folder, filename)

            shutil.copy(src_path, dst_path)

    copy_data('dog', 0, 1000, 'train/dogs')
    copy_data('cat', 0, 1000, 'train/cats')

    copy_data('dog', 1000, 1500, 'valid/dogs')
    copy_data('cat', 1000, 1500, 'valid/cats')

    copy_data('dog', 1500, 2000, 'test/dogs')
    copy_data('cat', 1500, 2000, 'test/cats')


def generator_basic():
    generator = tf.keras.preprocessing.image.ImageDataGenerator()

    train_flow = generator.flow_from_directory('dogs-vs-cats/small/train',
                                               target_size=[150, 150],
                                               batch_size=5,
                                               class_mode='binary')    # binary, sparse, categorical
    for i, data in enumerate(train_flow):
        print(type(data))                   # <class 'tuple'>
        print(len(data))                    # 2
        print(type(data[0]), type(data[1])) # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
        print(data[0].shape, data[1].shape) # (5, 150, 150, 3) (5,)
        print(data[1])
        # binary: [1. 0. 1. 0. 0.]
        # sparse: [0. 2. 0. 2. 2.]
        # categorical: [[0. 0. 1. 0.]
        #               [1. 0. 0. 0.]
        #               [0. 0. 1. 0.]
        #               [1. 0. 0. 0.]
        #               [0. 0. 1. 0.]]

        if i < 10:
            break


# make_folder_structure()
# make_small_datset()

generator_basic()
