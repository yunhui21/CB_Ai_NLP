# Day_04_03_konlpy.py
import konlpy
import collections

print(konlpy.corpus.kolaw.fileids())    # ['constitution.txt'] 헌법
print(konlpy.corpus.kobill.fileids())   # ['1809890.txt', '1809891.txt', '1809892.txt', '1809893.txt',...] 법안 제출 10개의 파일

# f = (konlpy.corpus.kolaw.open('constitution.txt'))
# print(f)
# print(f.read())
# f.close()

kolaw = konlpy.corpus.kobill.open('1809890.txt').read()
# print(kolaw)

# tokens = kolaw.split()
# print(tokens[:5])     # ['지방공무원법', '일부개정법률안', '(정의화의원', '대표발의', ')']

# Pos : part of speech
tagger = konlpy.tag.Hannanum()
pos = tagger.pos(kolaw)
print(pos[:10])
# java 설치 에러발생

# print(len(kolaw))
# print('morphs', len(pos))
# print('morphs', len(set(pos)))
#
# print('tokens', len(tokens))        #
# print('tokens', len(set(tokens)))   #tagging이 된 후에 vector를 해야 한다.
# print('-'*50)


#[('사람', 'N), ('가', 'y'), ...)

freq = collections.Counter(pos)
print(freq.most_common(10))

# 토큰 검색
indices = konlpy.utils.concordance('대한민국', kolaw)   # 단어상의 색인 hannanum에는 들어있음. kkoma는 없음.
print(indices)

# ('육아', 'nng')