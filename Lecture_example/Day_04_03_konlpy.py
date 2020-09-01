# Day_04_03_konlpy.py
import konlpy
import collections

print(konlpy.corpus.kolaw.fileids())    # ['constitution.txt']
print(konlpy.corpus.kobill.fileids())   # ['1809890.txt', ..., '1809899.txt']

# f = konlpy.corpus.kolaw.open('constitution.txt')
# print(f)
# print(f.read())
# f.close()

kolaw = konlpy.corpus.kobill.open('1809890.txt').read()
# print(kolaw)

tokens = kolaw.split()
# print(tokens[:5])

# pos : part of speech
# tagger = konlpy.tag.Hannanum()    # 이유는 모르지만, 로드가 안됨
tagger = konlpy.tag.Kkma()

pos = tagger.pos(kolaw)
print(pos[:10])

print(len(kolaw))

print('nmorphs :', len(pos))
print('nmorphs :', len(set(pos)))

print('tokens :', len(tokens))
print('tokens :', len(set(tokens)))
print('-' * 30)

# [('사람', 'N'), ('가', 'V'), ('ㄴ다', 'PR), ...]
freq = collections.Counter(pos)
print(freq.most_common(10))

indices = konlpy.utils.concordance('대한민국', kolaw)
print(indices)

# ('육아', 'NNG') => '육아/NNG'


