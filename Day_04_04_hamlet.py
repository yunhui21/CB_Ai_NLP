# Day_04_04_hamlet.py
import nltk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

# 문제
# 셰익스피어의 햄릿에 등장하는 주인공들의 출현 빈도로 막대 그래프로 그려 보세요(gutenberg)
# 햄릿, 그트루드, 올리어, 클로디어스, 레이터스, 플로니어스, 호레이스
# 1. 햄릿 설소 찾기
# 2. 등장 인물의 영어 림 찾기
# 3. 빈도 계산
# 4. 그래프 출력

# 1. 햄릿 설소 찾기
print(nltk.corpus.gutenberg.fileids())
# 'shakespeare-hamlet.txt'
hamlet = nltk.corpus.gutenberg.words('shakespeare-hamlet.txt')
hamlet = [ w.lower() for w in hamlet]

# 2. 등장 인물의 영어 림 찾기
actors = ['hamlet','claudius','gertrude','polonius','ophelia','horatio','laertes']
print([w in hamlet for w in actors])

# 3. 빈도 계산
# freqDist, counter
freq = nltk.FreqDist(hamlet)
print(freq.most_common(5))

freq_actors = [freq[w] for w in actors]     #딕셔너리로 쓰기위해 []사용
print(freq_actors)

# 4. 그래프 출력
# plt.barh(actors, freq_actors, color=colors.TABLEAU_COLORS)        #barh 수평
plt.bar(actors, freq_actors, color=colors.TABLEAU_COLORS)
plt.show()
