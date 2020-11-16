#Day_02_02_frequency.py
import nltk
import collections
import operator
import matplotlib.pyplot as plt

#nltk.download('webtext')
#nltk.download('stopwords')
#문제
#webtext  코퍼스에 어던 파일들이 있는지 보여주세요.
print(nltk.corpus.webtext.fileids())

#문제
#wine 데이터에 들어잇는 단어만 출력하세요.
wine =nltk.corpus.webtext.raw('wine.txt')
wine = wine.lower()
#정규표현식 r'\w+'
tokens = nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(wine)
print(tokens[:10])

print(nltk.corpus.stopwords.fileids())

stop_words = nltk.corpus.stopwords.words('english')

print(stop_words)
#문제
#토큰에서 stopwords와 한글자로 된 토큰을 삭제하세요.
#my_answer

# s_tokens = nltk.stem.LancasterStemmer()
# print([s_tokens.stem(w) for w in stop_words])

tokens = [t for t in tokens if t not in stop_words]
tokens = [t for t in tokens if len(t)>1]


print('='*30)
#문제
#토큰에 포함된 단어별 빈도를 딕셔너리에 저장하세요.
#{'the':120, 'hello':7, ...}
# 1번
# freq = {}
# for t in tokens:
#     if t in freq:
#         #freq[t] = freq[t] + 1
#         freq[t] += 1
#     else:
#         freq[t] = 1
# print(freq)

#2번
freq = {}
for t in tokens:
    freq[t] = freq.get(t, 0) + 1
print(freq)

#3번
#freq ={t:tokens.count(t) for t in tokens}
# freq ={t:tokens.count(t) for t in set(tokens)}
# print(freq)

#4번
# freq = collections.defaultdict(int)
# for t in tokens:
#     freq[t] += 1
# print(freq)

print(freq)

#문제
#빈도순으로 정렬된 리스트를 만드세요.
def second(t): #t:(key,value)
    return t[1]
#tops=sorted(freq.items())
#key값은 정의하고 싶은 함수를 넣으면 된다.operator
#tops=sorted(freq.items(), key= second)

#operator를 대신해서 lamda를 사용
#tops=sorted(freq.items(), key= lambda t:t[1], reverse=True)

#import operator /itmegetter는 2차 정렬 기능이 있음(0:key값, 1:value값)
tops=sorted(freq.items(), key= operator.itemgetter(1), reverse=True)
print(tops)
print('-'*30)

freq = nltk.FreqDist(tokens)
print(freq)
print(freq.N())
print(freq.most_common(10))

print(freq['good'])#특정단어의 빈도를 물어보는 작업 : good

#문제
#빈도 상위 10개를 막대 그래프로 그리세요.
print(*freq.most_common(3))
print(zip(*freq.most_common(3)))
print(list(zip(*freq.most_common(3))))
names, counts = list(zip(*freq.most_common(10)))

plt.bar(names, counts)
plt.show()

# plt.bar(*zip(*bar_plot.items(0)))
# plt.show()
#no1
# names, counts=[], []
# for k,c in freq.most_common(10):
#     names.append(k)
#     counts.append(c)
# idx = range(10)

#no2
# names =[k for k, _ in freq.most_common(10)]
# counts= [c for _, c in freq.most_common(10)]
# idx = range(10)

# plt.bar(idx, counts)
# plt.xticks(idx, names)
# plt.show()

