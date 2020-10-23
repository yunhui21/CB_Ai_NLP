# Day_34_02_tfds.py
import tensorflow_datasets as tfds

names = tfds.list_builders()
print(names)        # ['abstract_reasoning', 'accentdb', 'aeslc', ...
print(len(names))   # 224
print('-' * 30)

imdb_1 = tfds.load('imdb_reviews')
print(type(imdb_1))                 # <class 'dict'>
print(imdb_1.keys())                # dict_keys(['test', 'train', 'unsupervised'])
print(imdb_1)
# {'test': <PrefetchDataset shapes: {label: (), text: ()}, types: {label: tf.int64, text: tf.string}>,
#  'train': <PrefetchDataset shapes: {label: (), text: ()}, types: {label: tf.int64, text: tf.string}>,
#  'unsupervised': <PrefetchDataset shapes: {label: (), text: ()}, types: {label: tf.int64, text: tf.string}>}

print(type(imdb_1['test']))         # <class 'tensorflow.python.data.ops.dataset_ops.PrefetchDataset'>
print(imdb_1['test'])
print()

for row in imdb_1['test'].take(2):
    print(type(row), row.keys())    # <class 'dict'> dict_keys(['label', 'text'])
    print(row['label'], row['text'])
    # tf.Tensor(1, shape=(), dtype=int64) tf.Tensor(b"There are films that make careers. For George Romero, it was NIGHT OF THE LIVING DEAD; for Kevin Smith, CLERKS; for Robert Rodriguez, EL MARIACHI. Add to that list Onur Tukel's absolutely amazing DING-A-LING-LESS. Flawless film-making, and as assured and as professional as any of the aforementioned movies. I haven't laughed this hard since I saw THE FULL MONTY. (And, even then, I don't think I laughed quite this hard... So to speak.) Tukel's talent is considerable: DING-A-LING-LESS is so chock full of double entendres that one would have to sit down with a copy of this script and do a line-by-line examination of it to fully appreciate the, uh, breadth and width of it. Every shot is beautifully composed (a clear sign of a sure-handed director), and the performances all around are solid (there's none of the over-the-top scenery chewing one might've expected from a film like this). DING-A-LING-LESS is a film whose time has come.", shape=(), dtype=string)
    print(row['label'].numpy(), row['text'].numpy())
    # 1 b"There are films that make careers. For George Romero, it was NIGHT OF THE LIVING DEAD; for Kevin Smith, CLERKS; for Robert Rodriguez, EL MARIACHI. Add to that list Onur Tukel's absolutely amazing DING-A-LING-LESS. Flawless film-making, and as assured and as professional as any of the aforementioned movies. I haven't laughed this hard since I saw THE FULL MONTY. (And, even then, I don't think I laughed quite this hard... So to speak.) Tukel's talent is considerable: DING-A-LING-LESS is so chock full of double entendres that one would have to sit down with a copy of this script and do a line-by-line examination of it to fully appreciate the, uh, breadth and width of it. Every shot is beautifully composed (a clear sign of a sure-handed director), and the performances all around are solid (there's none of the over-the-top scenery chewing one might've expected from a film like this). DING-A-LING-LESS is a film whose time has come."
print()

for row in imdb_1['test'].take(2).as_numpy_iterator():
    print(row)
    # {'label': 1, 'text': b"There are films that make careers. For George Romero, it was NIGHT OF THE LIVING DEAD; for Kevin Smith, CLERKS; for Robert Rodriguez, EL MARIACHI. Add to that list Onur Tukel's absolutely amazing DING-A-LING-LESS. Flawless film-making, and as assured and as professional as any of the aforementioned movies. I haven't laughed this hard since I saw THE FULL MONTY. (And, even then, I don't think I laughed quite this hard... So to speak.) Tukel's talent is considerable: DING-A-LING-LESS is so chock full of double entendres that one would have to sit down with a copy of this script and do a line-by-line examination of it to fully appreciate the, uh, breadth and width of it. Every shot is beautifully composed (a clear sign of a sure-handed director), and the performances all around are solid (there's none of the over-the-top scenery chewing one might've expected from a film like this). DING-A-LING-LESS is a film whose time has come."}
    print(row['label'], row['text'])
    # 1 b"A blackly comic tale of a down-trodden priest, Nazarin showcases the economy that Luis Bunuel was able to achieve in being able to tell a deeply humanist fable with a minimum of fuss. As an output from his Mexican era of film making, it was an invaluable talent to possess, with little money and extremely tight schedules. Nazarin, however, surpasses many of Bunuel's previous Mexican films in terms of the acting (Francisco Rabal is excellent), narrative and theme.<br /><br />The theme, interestingly, is something that was explored again in Viridiana, made three years later in Spain. It concerns the individual's struggle for humanity and altruism amongst a society that rejects any notion of virtue. Father Nazarin, however, is portrayed more sympathetically than Sister Viridiana. Whereas the latter seems to choose charity because she wishes to atone for her (perceived) sins, Nazarin's whole existence and reason for being seems to be to help others, whether they (or we) like it or not. The film's last scenes, in which he casts doubt on his behaviour and, in a split second, has to choose between the life he has been leading or the conventional life that is expected of a priest, are so emotional because they concern his moral integrity and we are never quite sure whether it remains intact or not.<br /><br />This is a remarkable film and I would urge anyone interested in classic cinema to seek it out. It is one of Bunuel's most moving films, and encapsulates many of his obsessions: frustrated desire, mad love, religious hypocrisy etc. In my view 'Nazarin' is second only to 'The Exterminating Angel', in terms of his Mexican movies, and is certainly near the top of the list of Bunuel's total filmic output."
print('-' * 30)

imdb_2, info = tfds.load('imdb_reviews', with_info=True)
print(info)

# 문제
# train과 test의 데이터 갯수를 출력하세요
print(info.splits)
# {'test': <tfds.core.SplitInfo num_examples=25000>, 'train': <tfds.core.SplitInfo num_examples=25000>, 'unsupervised': <tfds.core.SplitInfo num_examples=50000>}
print(info.splits.keys())       # dict_keys(['test', 'train', 'unsupervised'])
print(info.splits['train'])     # <tfds.core.SplitInfo num_examples=25000>
print(info.splits['train'].num_examples)    # 25000
print(info.splits['test'].num_examples)     # 25000
print('-' * 30)

imdb_3 = tfds.load('imdb_reviews', as_supervised=True)
print(imdb_3)
# {'test': <PrefetchDataset shapes: ((), ()), types: (tf.string, tf.int64)>, 'train': <PrefetchDataset shapes: ((), ()), types: (tf.string, tf.int64)>, 'unsupervised': <PrefetchDataset shapes: ((), ()), types: (tf.string, tf.int64)>}
print(imdb_3.keys())        # dict_keys(['test', 'train', 'unsupervised'])
print(imdb_3['test'])
# <PrefetchDataset shapes: ((), ()), types: (tf.string, tf.int64)>

for row in imdb_3['test'].take(2):
    print(type(row), row)
    # <class 'tuple'> (<tf.Tensor: shape=(), dtype=string, numpy=b"There are films that make careers. For George Romero, it was NIGHT OF THE LIVING DEAD; for Kevin Smith, CLERKS; for Robert Rodriguez, EL MARIACHI. Add to that list Onur Tukel's absolutely amazing DING-A-LING-LESS. Flawless film-making, and as assured and as professional as any of the aforementioned movies. I haven't laughed this hard since I saw THE FULL MONTY. (And, even then, I don't think I laughed quite this hard... So to speak.) Tukel's talent is considerable: DING-A-LING-LESS is so chock full of double entendres that one would have to sit down with a copy of this script and do a line-by-line examination of it to fully appreciate the, uh, breadth and width of it. Every shot is beautifully composed (a clear sign of a sure-handed director), and the performances all around are solid (there's none of the over-the-top scenery chewing one might've expected from a film like this). DING-A-LING-LESS is a film whose time has come.">, <tf.Tensor: shape=(), dtype=int64, numpy=1>)
print()

for text, label in imdb_3['test'].take(2):
    print(label, text)
print('-' * 30)

imdb_4 = tfds.load('imdb_reviews', split=['train', 'test'])
print(type(imdb_4), len(imdb_4))

train_set, test_set = tfds.load('imdb_reviews',
                                as_supervised=True,
                                split=['train', 'test'])
print(type(train_set))
# <class 'tensorflow.python.data.ops.dataset_ops.PrefetchDataset'>

for text, label in train_set.take(2):
    print(label, text)
print('-' * 30)

train_set, valid_set, test_set = tfds.load(
    'imdb_reviews', as_supervised=True,
    split=['train[:60%]', 'train[60%:]', 'test'])
print(train_set.cardinality())  # tf.Tensor(15000, shape=(), dtype=int64)
print(valid_set.cardinality())  # tf.Tensor(10000, shape=(), dtype=int64)
print(test_set.cardinality())   # tf.Tensor(25000, shape=(), dtype=int64)
