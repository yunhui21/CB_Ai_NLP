#Day_04_02_word2vec.py
import gensim
import nltk

print(nltk.corpus.movie_reviews.fileids())
# ['neg/cv000_29416.txt', 'neg/cv001_19502.txt', 'neg/cv002_17424.txt',...]

# print(nltk.corpus.movie_reviews.raw('neg/cv000_29416.txt'))      # 문자열, str
# print(nltk.corpus.movie_reviews.words('neg/cv000_29416.txt'))    # 1차원 리스트, []
# print(nltk.corpus.movie_reviews.sents('neg/cv000_29416.txt'))    # 2차원 리스트, []

sents = nltk.corpus.movie_reviews.sents()
model = gensim.models.Word2Vec(sents)

print(model)
# Word2Vec(vocab=14794, size=100, alpha=0.025)

# 같은 방향(1), 반대방향 (-1), 직교(0)
# word2vec안의 단어를 사용해야한다.
print(model.wv.similarity('smile',  'space'))      # 0.49503884
print(model.wv.similarity('man',    'women'))      # 0.28579506
print(model.wv.similarity('sky',    'earth'))      # 0.5859858

print(model.wv.most_similar('men'))
'''
[('women',    0.8841222524642944),  ('girls',    0.821674644947052),  ('humans',   0.8073922991752625), 
 ('cops',     0.7765034437179565),  ('boys',     0.7671017646789551), ('fighting', 0.764306902885437), 
 ('faces',    0.7524381875991821),  ('children', 0.7504572868347168), ('heroes',   0.739061713218689), 
 ('soldiers', 0.738467812538147)]
'''

print('men' in model.wv)
