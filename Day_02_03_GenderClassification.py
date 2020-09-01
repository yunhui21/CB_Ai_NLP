#Day_02_03_GenderClassification.py
import nltk
import random

# nltk.download('names')
#문제
#이름을 이용해서 성별을 예측하세요.

#print(nltk.corpus.names.fileids()) #['female.txt', 'male.txt']
#print(nltk.corpus.names.raw('male.txt')[:100])
# 문제
# make_labled_name 함수를 만드세요.
#아래와 같은 형식으로 반환해야 한다.
# [
# ('kim','male')
# ('dark','male')
# ('lee','female')
# ('nam','male')
# ... ]

#my_answer
# gender_male = nltk.corpus.names.words('male.txt')
# gender_male = lower()
# male_token  = nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(gender_male)
# stop_words  = nltk.corpus.stopwords.words('english')
# male_tokens = [t for t in male_token if t not in stop_words]
# male_tokens = [t for t in male_token if len(t)>1]
#
# gender_female = nltk.corpus.names.words('female.txt')
# gender_female = lower()
# female_token  = nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(gender_female)
# stop_words    = nltk.corpus.stopwords.words('english')
# female_tokens = [t for t in female_token if t not in stop_words]
# female_tokens = [t for t in female_token if len(t)>1]

def make_labled_names():
    males = [(n.strip(),'male') for n in nltk.corpus.names.words('male.txt')]
    females = [(n.strip(),'female') for n in nltk.corpus.names.words('female.txt')]

    print(len(males), len(females))

    names= males + females#list
    print(names)

    random.shuffle(names)
    print(*names[:5], sep='\n')

    return names


def make_feature_1(name):
    feature ={'last_letter' : name[-1]}
    return feature


def make_feature_2(name):
    feature ={
        'first_letter': name[0],
        'last_letter' : name[-1]
    }
    for ch in 'abcdefghijklmnopqrstuvwxyz': #26개의 캐릭터를 갖고옴
                                            # 예 : a가 3개일때 여자일까 남자일가를 상상.
        feature['count_({})', format(ch)] = name.count(ch)
        feature['has_({})', format(ch)] = (ch in name)

    return feature


def make_feature_3(name):#소문자를 count하는 요소에 맞추기 위해 lower()
    name = name.lower()
    feature ={
        'first_letter': name[0],
        'last_letter' : name[-1]
    }
    for ch in 'abcdefghijklmnopqrstuvwxyz': #26개의 캐릭터를 갖고옴
                                            #예 : a가 3개일때 여자일까 남자일가를 상상.
        feature['count_({})', format(ch)] = name.count(ch)
        feature['has_({})', format(ch)] = (ch in name)

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
        'suffix_2': name[-2:]#마지막 글자2개를 동시에 본다.:
    }
    return feature

def make_feature_data(names, make_feature):
    return [(make_feature(name), gender)for name, gender in names]

# 문제
# 남자와 여자의 이름에 포함되지 안은 알파벳을 찾으세요.
# namee, data, last_letter, [for i in ]
# 이름의 누락된 마지막 글자는 찿기
# make_labeld_names에서 표현
def show_omitted(names):
    men, women = set(), set()  # 셋을 활용
    for name, gender in names:  #
        if gender == 'male':  # 성별이 남자면
            men.add(name[-1])
        else:
            women.add(name[-1])  # 여자인 경우
        print(name)

    print(sorted(men))
    print(sorted(women))
    print(men - women)  # {'c'}
    print(women - men)  # set{}공백을 표현

#문제
#잘못 예측한 이름을 알려주세요.
#남자 : Dan, male
#여자 : alf, female
#(nltk.classify.accuracy(clf, train_set)) train_set의 값을 검토해라
# 매개변수는 무엇으로 볼것인가> classf
def check_mismatch(clf, test_set, make_feature):
    men, women = [], []
    for name, gender in test_set:
        #print(feature, gender)
        pred = clf.classify(make_feature_3(name))
        if  pred == 'gender':
            continue
        # print(name)
        if pred =='male':
            men.append(name)
        else:
            women.append(name)

    print('남자:', *sorted(men))
    print('여자:', *sorted(women))


# def gender_basic():
#     names = make_labled_names()
#     #data = make_feature_data(names, make_feature_1)
#     # data = make_feature_data(names, make_feature_2)
#     # data = make_feature_data(names, make_feature_3)
#     # data = make_feature_data(names, make_feature_4)
#     data = make_feature_data(names, make_feature_5)
#
#     print(data[:3])
#
#     # [({'first_letter': 'K', 'last_letter': 'i',
#     #    ('count_({})', 'a'): 1, ('has_({})', 'a'): True,
#     #    ('count_({})', 'b'): 0, ('has_({})', 'b'): False,
#     #    ('count_({})', 'c'): 0, ('has_({})', 'c'): False,
#     #    ('count_({})', 'd'): 0, ('has_({})', 'd'): False},---)]
#
#     #question
#     #검사에 1000개, 학습에 나머지를 사용하도록 분할하세요.
#     #split[:1000], [1000:]
#
#     # train_set, test_set = data.[:-1000], data[-1000:]
#     train_set, test_set = data[1000:], data[:1000]#shuffl을 적용했으므로
#
#     clf = nltk.NaiveBayesClassifier.train(train_set)
#     print(nltk.classify.accuracy(clf, train_set))
#     #name을 피쳐로 변환해서 넣어야 한다.
#     print(clf.classify(make_feature_3('Trinity')))
#     print(clf.classify(make_feature_3('Neo')))
#
#
#     # clf.show_most_informative_features(5)
#     # clf.show_most_informative_features()#값을 지정하지않으면 default값을 반환 10개
#     # clf.show_most_informative_features(50)
#     # check_mismatch(clf, names[:1000], make_feature_3)


def gender_comparison():
    names = make_labled_names()

    for make_feature in [make_feature_1, make_feature_2, make_feature_3, make_feature_4, make_feature_5]:

        data = make_feature_data(names, make_feature)
        train_set, test_set = data[1000:], data[:1000]  # shuffl을 적용했으므로

        clf = nltk.NaiveBayesClassifier.train(train_set)
        print(nltk.classify.accuracy(clf, train_set))

# def gender_comparison():
#     names = make_labled_names()
#     # data = make_feature_data(names, make_feature_1)
#     # data = make_feature_data(names, make_feature_2)
#     # data = make_feature_data(names, make_feature_3)
#     # data = make_feature_data(names, make_feature_4)
#     data = make_feature_data(names, make_feature_5)
#
#     print(data[:3])
#
#     # [({'first_letter': 'K', 'last_letter': 'i',
#     #    ('count_({})', 'a'): 1, ('has_({})', 'a'): True,
#     #    ('count_({})', 'b'): 0, ('has_({})', 'b'): False,
#     #    ('count_({})', 'c'): 0, ('has_({})', 'c'): False,
#     #    ('count_({})', 'd'): 0, ('has_({})', 'd'): False},---)]
#
#     # question
#     # 검사에 1000개, 학습에 나머지를 사용하도록 분할하세요.
#     # split[:1000], [1000:]
#
#     # train_set, test_set = data.[:-1000], data[-1000:]
#     train_set, test_set = data[1000:], data[:1000]  # shuffl을 적용했으므로
#
#     clf = nltk.NaiveBayesClassifier.train(train_set)
#     print(nltk.classify.accuracy(clf, train_set))
#     # name을 피쳐로 변환해서 넣어야 한다.
#     print(clf.classify(make_feature_3('Trinity')))
#     print(clf.classify(make_feature_3('Neo')))
#
#     # clf.show_most_informative_features(5)
#     # clf.show_most_informative_features()#값을 지정하지않으면 default값을 반환 10개
#     # clf.show_most_informative_features(50)
#     # check_mismatch(clf, names[:1000], make_feature_3)

#
#피쳐로 만들어야 하는 작업을 진행해야한다.
#data를 만드는 과정
# [
# ('kim','male')
# ('dark','male')
# ('lee','female')
# ('nam','male')
# ... ]

