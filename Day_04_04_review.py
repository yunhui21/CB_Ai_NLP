# Day_04_04_review.py
import nltk
import string
# 문제
# 섹스피어의 햄릿에 등장하는 주인공들의 출현 빈도를 막대 그래프로 그려 보세요.(gutenberg)
# 거크루드, 오필리어,

print(nltk.corpus.gutenberg.fileids())
'''['austen-emma.txt', 'austen-persuasion.txt', 'austen-sense.txt', 'bible-kjv.txt', 
    'blake-poems.txt', 'bryant-stories.txt', 'burgess-busterbrown.txt', 'carroll-alice.txt', 
    'chesterton-ball.txt', 'chesterton-brown.txt', 'chesterton-thursday.txt', 'edgeworth-parents.txt', 
    'melville-moby_dick.txt', 'milton-paradise.txt', 'shakespeare-caesar.txt', 'shakespeare-hamlet.txt', 
    'shakespeare-macbeth.txt', 'whitman-leaves.txt']'''
print('-'*50)
# hamlet = nltk.corpus.gutenberg.raw('shakespeare-hamlet.txt')
# print(hamlet)
# print(type(hamlet))     #<class 'str'>

print(nltk.corpus.gutenberg.words('shakespeare-hamlet.txt'))
print('-'*50)



# text = nltk.corpus.gutenberg.raw('shakespeare-hamlet.txt')
# text = text[:]
# print(text)
# print('-'*50)
#
#
# print(nltk.tokenize.simple.SpaceTokenizer().tokenize(text))
# print(nltk.tokenize.sent_tokenize(text))
# print('-'*50)

#
# for sent in nltk.tokenize.sent_tokenize(text):
#     print(sent)
#     print('------------------')
#
# print(nltk.tokenize.RegexpTokenizer(r'[a-zA-Z]+').tokenize(text))
# print(nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(text))

