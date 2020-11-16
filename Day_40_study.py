import random
import string
import numpy as np
# 01
a = ['Guest', 'Detroit', 'Woodlawn', 'Cemetary', 'People']
print([i for i in a])
print([len(i) for i in a])
print(max([len(i) for i in a]))

# 02
print([i for i in range(10)])
print([random.randint(1, 100) for i in range(10)])

# 03
b = [31, 46, 71]
y = reversed(b)
print(y)

# 04
alp = list(string.ascii_lowercase)
print(alp)
letters = [['a', 'b', 'c', 'd', 'e', 'f'],
           ['g', 'h', 'i', 'j', 'k', 'l'],
           ['m', 'n', 'o', 'p', 'q', 'r'],
           ['s', 't', 'u', 'v', 'w', 'x']]
for s in letters:
    # print(s)
    for c in s:
        print(c, end='')
print()

# print([[c for c in s] for s in letters])
print([e for array in letters for e in array])

# 05
t = ([random.randint(1,10) for i in range(10)])
print(t)
# print(*t, sep='\n')
# print([i for i in t])
# print([[j for j in i if j % 2] for i in t])
# print([sum[[j for j in i if j % 2]] for i in t])
# print(max([sum[[j for j in i if j % 2]] for i in t]))
# print(np.argumax([sum[[j for j in i if j % 2]]for i in t]))

# frequency

import nltk
import collections


print(nltk.corpus.webtext.fileids())
# ['firefox.txt', 'grail.txt', 'overheard.txt', 'pirates.txt', 'singles.txt', 'wine.txt']
wine = nltk.corpus.webtext.raw('wine.txt')
wine = wine.lower()
token = nltk.tokenize.RegexpTokenizer('r\w+').tokenize(wine)
print(token[:10])
print(nltk.corpus.stopwords.fileids())
# ['arabic', 'azerbaijani', 'danish', 'dutch', 'english', 'finnish', 'french', 'g
# erman', 'greek', 'hungarian', 'indonesian', 'italian', 'kazakh', 'nepali',
# 'norwegian', 'portuguese', 'romanian', 'russian', 'slovene', 'spanish', 'swedish', 'tajik', 'turkish']
stop_words = nltk.corpus.stopwords.words('english')
print(stop_words)

# 문제
# 토큰에서 stopwords와 한글자로 된 토큰을 삭제하세요.
# stemming
# stokens = nltk.stem.LancasterStemmer()
# print([stokens.stem(w) for w in wine])
tokens = [t for t in tokens if t not in stop_words]
tokens = [t for t in tokens if len(t) > 1]

# 딕셔너리에 저장
# freq = {}
# for t in tokens:
#     if t in freq:
#         freq[t] += 1
#     else:
#         freq[t] = 1

freq = {}
for t in tokens:
    freq[t] = freq.get(t, 0) + 1

# freq = {t:tokens.count(t) for t in tokens}
# freq = {t:tokens.count(t) for t in set(tokens)}

# freq = collections.defaultdict(int)
# for t in tokens:
#     freq[t] += 1

print(freq)

def second(t):
    return t[1]




