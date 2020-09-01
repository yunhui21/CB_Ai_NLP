# Day_02_03_GenderClassification.py
import nltk
import random

# 문제
# 이름을 이용해서 성별을 예측하세요
# nltk.download('names')

# 문제
# make_labeled_names 함수를 만드세요
# 아래와 같은 형식으로 반환해야 합니다
# [
# ('kim', 'male'),
# ('park', 'male'),
# ('lee', 'female'),
# ('nam', 'male'),
# ...]

# print(nltk.corpus.names.fileids())  # ['female.txt', 'male.txt']
# print(nltk.corpus.names.raw('male.txt')[:100])
def make_labeled_names():
    males = [(n.strip(), 'male') for n in nltk.corpus.names.words('male.txt')]
    females = [(n.strip(), 'female') for n in nltk.corpus.names.words('female.txt')]

    print(len(males), len(females))     # 2943 5001

    names = males + females
    print(len(names))                   # 7944

    random.shuffle(names)
    print(*names[:5], sep='\n')

    return names


def make_feature_1(name):
    feature = {'last_letter': name[-1]}
    return feature


def make_feature_2(name):
    feature = {
        'first_letter': name[0],
        'last_letter': name[-1]
    }
    for ch in 'abcdefghijklmnopqrstuvwxyz':
        feature['count_({})'.format(ch)] = name.count(ch)
        feature['has_({})'.format(ch)] = (ch in name)

    return feature


def make_feature_3(name):
    name = name.lower()
    feature = {
        'first_letter': name[0],
        'last_letter': name[-1]
    }
    for ch in 'abcdefghijklmnopqrstuvwxyz':
        feature['count_({})'.format(ch)] = name.count(ch)
        feature['has_({})'.format(ch)] = (ch in name)

    return feature


def make_feature_4(name):
    feature = {
        'suffix_1': name[-1],
        'suffix_2': name[-2]
    }
    return feature


def make_feature_5(name):
    feature = {
        'suffix_1': name[-1],
        'suffix_2': name[-2:]
    }
    return feature


def make_feature_data(names, make_feature):
    return [(make_feature(name), gender) for name, gender in names]


# 문제
# 남자와 여자 이름에 포함되지 않은 알파벳을 찾으세요
def show_omitted(names):
    men, women = set(), set()
    for name, gender in names:
        if gender == 'male':
            men.add(name[-1])
        else:
            women.add(name[-1])
        print(name)

    print(sorted(men))
    print(sorted(women))
    print(men - women)  # {'c'}
    print(women - men)  # set()


# 문제
# 잘못 예측한 이름을 알려주세요
# 남자 : ben, mark
# 여자 : ali, jennie
def check_mismatch(clf, test_set, make_feature):
    men, women = [], []
    for name, gender in test_set:
        # print(name, gender)
        pred = clf.classify(make_feature(name))
        if pred == gender:
            continue

        # print(name)
        if pred == 'male':
            men.append(name)
        else:
            women.append(name)

    print('남자 :', *sorted(men))
    print('여자 :', *sorted(women))


def gender_basic():
    names = make_labeled_names()
    # data = make_feature_data(names, make_feature_1)
    # data = make_feature_data(names, make_feature_2)
    # data = make_feature_data(names, make_feature_3)
    # data = make_feature_data(names, make_feature_4)
    data = make_feature_data(names, make_feature_5)
    print(data[:3])
    # [({'first_letter': 'G', 'last_letter': 'y',
    # 'count_(a)': 1, 'has_(a)': True, 'count_(b)': 0, 'has_(b)': False,
    # 'count_(c)': 0, 'has_(c)': False, 'count_(d)': 0, 'has_(d)': False,
    # ...]

    # 문제
    # 검사에 1000개, 학습에 나머지를 사용하도록 분할하세요
    # train_set, test_set = data[:-1000], data[-1000:]
    train_set, test_set = data[1000:], data[:1000]

    clf = nltk.NaiveBayesClassifier.train(train_set)
    print(nltk.classify.accuracy(clf, test_set))

    print(clf.classify(make_feature_3('Trinity')))
    print(clf.classify(make_feature_3('Neo')))

    # clf.show_most_informative_features(5)
    # clf.show_most_informative_features()    # 10
    # clf.show_most_informative_features(50)

    # show_omitted(names)
    # check_mismatch(clf, names[:1000], make_feature_3)


def gender_comparison():
    names = make_labeled_names()

    for make_feature in [make_feature_1, make_feature_2, make_feature_3, make_feature_4, make_feature_5]:
        data = make_feature_data(names, make_feature)
        train_set, test_set = data[1000:], data[:1000]

        clf = nltk.NaiveBayesClassifier.train(train_set)
        print(nltk.classify.accuracy(clf, test_set))


# gender_basic()
gender_comparison()



