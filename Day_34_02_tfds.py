# Day_34_02_tfds.py
import tensorflow_datasets as tfds

names = tfds.list_builders()
print(names)        # ['abstract_reasoning', 'accentdb', 'aeslc', 'aflw2k3d',...
print(len(names))   # 224

imdb_1 = tfds.load('imdb_reviews')
print(imdb_1)
print(type(imdb_1)) # <class 'dict'>
print(imdb_1.keys()) # dict_keys(['test', 'train', 'unsupervised'])
'''
prepatch : 연산전에 갖고오는 데이터
{'test': <PrefetchDataset shapes: {label: (), text: ()}, types: {label: tf.int64, text: tf.string}>, 
'train': <PrefetchDataset shapes: {label: (), text: ()}, types: {label: tf.int64, text: tf.string}>, 
'unsupervised': <PrefetchDataset shapes: {label: (), text: ()}, types: {label: tf.int64, text: tf.string}>}
'''
print(type(imdb_1['test'])) # <class 'tensorflow.python.data.ops.dataset_ops.PrefetchDataset'>
print(imdb_1['test']) # <PrefetchDataset shapes: {label: (), text: ()}, types: {label: tf.int64, text: tf.string}>
print('-'*50)


# dataset이므로
for row in imdb_1['test'].take(1):
    print(type(row), row.keys()) # <class 'dict'> dict_keys(['label', 'text'])
    print(row['label'], row['text']) # tf.Tensor(1, shape=(), dtype=int64) tf.Tensor(b"There are films..
    print(row['label'].numpy(), row['text'].numpy()) # 1 b"There are films that make careers. For George Romero, it was N..
print()


for row in imdb_1['test'].take(2).as_numpy_iterator():
    print(row)
    print(row['label'], row['text'])
print()

imdb_2, info = tfds.load('imdb_reviews', with_info=True)
print(info)
print('-'*50)

# 문제
# train과 test의 데이터를 출력하세요.
print(info.splits) # {'test': <tfds.core.SplitInfo num_examples=25000>, 'train': <tfds.core.SplitInfo num_examples=25000>, 'unsupervised': <tfds.core.SplitInfo num_examples=50000>}
print(info.splits.keys()) # dict_keys(['test', 'train', 'unsupervised'])
print(info.splits['train']) # <tfds.core.SplitInfo num_examples=25000>
print(info.splits['train'].num_examples) # 25000
print(info.splits['test'].num_examples) # 25000

# print(imdb_2['test']) # <PrefetchDataset shapes: {label: (), text: ()}, types: {label: tf.int64, text: tf.string}>
# for row in imdb_2['test'].take(1):
#     print(row['label'].numpy(), row['text'].numpy())
# # 1 b"There are films that make careers. For George Romero, it was
#
# print(imdb_2['train']) # <PrefetchDataset shapes: {label: (), text: ()}, types: {label: tf.int64, text: tf.string}>
# for row in imdb_2['train'].take(1):
#     print(row['label'].numpy(), row['text'].numpy())
# # 0 b"This was an absolutely terrible movie. Don't be lured in by...


imdb_3 = tfds.load('imdb_reviews', as_supervised=True)
print(imdb_3) # {'test': <PrefetchDataset shapes: ((), ()), types: (tf.string, tf.int64)>,
print(imdb_3.keys()) # dict_keys(['test', 'train', 'unsupervised'])
print(imdb_3['test'])  # <PrefetchDataset shapes: ((), ()), types: (tf.string, tf.int64)>

for row in imdb_3 ['test'].take(2):
    print(type(row), row)
print() # <class 'tuple'> (<tf.Tensor: shape=()

for text, label in imdb_3['test'].take(2):
    print(label.numpy, text)
print() # <class 'tuple'> (<tf.Tensor: shape=()
print('-'*50)

imdb_4 = tfds.load('imdb_reviews', split=['train', 'test'])
print(type(imdb_4), len(imdb_4))

train_set, test_set = tfds.load('imdb_reviews',
                                as_supervised=True,
                                split=['train', 'test'])
print(type(train_set))

for text, label in train_set.take(2):
    print(label, text)
print('-'*50)

train_set, valid_set, test_set = tfds.load('imdb_reviews',
                                as_supervised=True,
                                split=['train[:60%]', 'train[60%:]', 'test']) # Dataset에서만 사용 가능한 문법

print(train_set.cardinality(), test_set.cardinality(), test_set.cardinality())
# tf.Tensor(25000, shape=(), dtype=int64) tf.Tensor(25000, shape=(), dtype=int64) tf.Tensor(25000, shape=(), dtype=int64)
# tf.Tensor(15000, shape=(), dtype=int64) tf.Tensor(25000, shape=(), dtype=int64) tf.Tensor(25000, shape=(), dtype=int64)