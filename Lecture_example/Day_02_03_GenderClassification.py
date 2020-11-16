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

# gender_male = nltk.corpus.names.words('male.txt')
# gender_male = lower()
# male_token = nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(gender_male)
# stop_words = nltk.corpus.stopwords.words('english')
# male_tokens = [t for t in male_token if t not in stop_words]
# male_tokens = [t for t in male_token if len(t)>1]
#
#
#
#
#
# gender_female = nltk.corpus.names.words('female.txt')
# gender_female = lower()
# female_token = nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(gender_female)
# stop_words = nltk.corpus.stopwords.words('english')
# female_tokens = [t for t in female_token if t not in stop_words]
# female_tokens = [t for t in female_token if len(t)>1]

def make_labled_names():
    males = [(n,'male') for n in nltk.corpus.names.words('male.txt')]
    females = [(n,'female') for n in nltk.corpus.names.words('female.txt')]

    print(len(males), len(females))

    names= males + females#list
    print(names)

    random.shuffle(names)
    print(*names[:5], sep='\n')

    return names

def make_feature(name):
    features ={'last_letter' : name[0][-1]}
    return features

def make_feature_data(names):
    return [(make_feature, gender)for name, gender in names]


names = make_labled_names()
make_feature_data()

clf = nltk.NaiveBayesClassifier.train(data)
print(nltk.classify.accuracy(clf, data))
#피쳐로 만들어야 하는 작업을 진행해야한다.
#data를 만드는 과정
# [
# ('kim','male')
# ('dark','male')
# ('lee','female')
# ('nam','male')
# ... ]
