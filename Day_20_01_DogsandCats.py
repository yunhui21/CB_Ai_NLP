# Day_20_01_DogsandCats.py
# import tensorflow as tf
# import numpy as np
# from PIL import Image
import os
import shutil
import tensorflow as tf
'''
print('a' * 3)
print(['a'] * 3)
# print({'a'} * 3) Error
print([{'a': 12}] * 3)


[{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': 1 # 4번활용
  }] * (num_units - 1) + [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': stride  # 2일때도 있다 1번은 다른값으로 총 5회가 되도록 한다.
  }]
'''
# 문제
# 이와 같은 구조로 폴더를 만들고, 만약 폴더가 있으면 만들지 않는다.

# 문제
# train 폴더에 1000개, valid 폴더에 500개, test 폴더에 500개

def make_folder_structure():
    def make_folder_if_not_exist(dir_path):
        if os.path.exists(dir_path):
            # os.mkdir(dir_path) # 상위폴더 생성후 하위 폴더 생성
            os.makedirs(dir_path) # 상위폴더 생성없이도 하위폴더 생성

    # make_folder_if_not_exist('data/dogs-vs-cats/samll')
    #
    # make_folder_if_not_exist('data/dogs-vs-cats/samll/train')
    # make_folder_if_not_exist('data/dogs-vs-cats/samll/test')
    # make_folder_if_not_exist('data/dogs-vs-cats/samll/valid')

    make_folder_if_not_exist('ddogs-vs-cats/smll/train/dogs')
    make_folder_if_not_exist('dogs-vs-cats/smll/train/cats')
    make_folder_if_not_exist('dogs-vs-cats/smll/test/dogs')
    make_folder_if_not_exist('dogs-vs-cats/smll/test/cats')
    make_folder_if_not_exist('dogs-vs-cats/smll/valid/dogs')
    make_folder_if_not_exist('dogs-vs-cats/smll/valid/cats')

    # base_dir = 'data/dogs-vs-cats/samll/'
    # if (os.path.isdir(os.path.join(base_dir, 'train', 'test',' valid')) == False)
    #     os.mkdir('train', 'test', 'valid')
    # # dir_path = os.path.join(base_dir, 'train', 'test',' valid')
    # base_train = 'data/dogs-vs-cats/samll/train'
    # dir_path = os.path.join(base_train, 'cats', 'dogs')
    # base_test = 'data/dogs-vs-cats/samll/test'
    # dir_path = os.path.join(base_test, 'cats', 'dogs')
    # base_valid = 'data/dogs-vs-cats/samll/valid'
    # dir_path = os.path.join(base_test, 'cats', 'dogs')

def small_dataset_0():
    train_dir = 'data/dogs-vs-cats/train'
    fnames = ['cats.{}.jpg'.format(i) for i in range(500)]
    for fname in fnames:
        src = os.path.join(train_dir, fname)
        dst = os.path.join('data/dogs-vs-cats/small/train/cats', fname)
        shutil.copyfile(src, dst)

    fnames = ['dogs.{}.jpg'.format(i) for i in range(500)]
    for fname in fnames:
        src = os.path.join(train_dir, fname)
        dst = os.path.join('data/dogs-vs-cats/small/train/dogs', fname)
        shutil.copyfile(src, dst)


    fnames = ['cats.{}.jpg'.format(i) for i in range(250)]
    for fname in fnames:
        src = os.path.join(train_dir, fname)
        dst = os.path.join('data/dogs-vs-cats/small/test/cats', fname)
        shutil.copyfile(src, dst)
    for fname in fnames:
        src = os.path.join(train_dir, fname)
        dst = os.path.join('data/dogs-vs-cats/small/valid/cats', fname)
        shutil.copyfile(src, dst)

    fnames = ['dogs.{}.jpg'.format(i) for i in range(250)]
    for fname in fnames:
        src = os.path.join(train_dir, fname)
        dst = os.path.join('data/dogs-vs-cats/small/test/dogs', fname)
        shutil.copyfile(src, dst)
    for fname in fnames:
        src = os.path.join(train_dir, fname)
        dst = os.path.join('data/dogs-vs-cats/small/valid/dogs', fname)
        shutil.copyfile(src, dst)

def small_dataset_():
    def copy_data(animal, start, end, target):
        dst_folder = os.path.join('data/dogs-vs-cats/small', target)

        for i in range(start, end):
            filename = '{}.{}.jpg'.format(animal, i)
            # print(filename)
            src_path = os.path.join('data/dogs-vs-cats/train', filename)
            dst_path = os.path.join(dst_folder, filename)

            shutil.copy(src_path, dst_path)

    copy_data('dog', 0, 1000, 'train/dogs' )
    copy_data('cat', 0, 1000, 'train/cats' )

    copy_data('dog', 1000, 1500, 'test/dogs' )
    copy_data('cat', 1000, 1500, 'test/cats' )

    copy_data('dog', 1500, 2000, 'valid/dogs' )
    copy_data('cat', 1500, 2000, 'valid/cats' )

def generator_basic():
    generator  = tf.keras.preprocessing.image.ImageDataGenerator()

    train_flow = generator.flow_from_directory('data/dogs-vs-cats/small/train',
                                               target_size=[150, 150],
                                               batch_size= 5,
                                               class_mode='categorical')

    for i, data in enumerate(train_flow):
        print(type(data))                   # <class 'tuple'>
        print(len(data))                    # 2
        print(type(data[0]), type(data[1])) # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
        print(data[0].shape, data[1].shape) # (5, 150, 150, 3) (5,)
        print(data[1])                      # [0. 1. 1. 0. 0.]
                                            # categorical
                                            # [[1. 0. 0. 0.],[0. 0. 1. 0.],[0. 0. 1. 0.],[1. 0. 0. 0.],[1. 0. 0. 0.]]
        if i < 10:
            break

# make_folder_structure()
# small_dataset_0()
# small_dataset_()
generator_basic()


