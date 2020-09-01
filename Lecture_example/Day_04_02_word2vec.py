# Day_04_02_word2vec.py
import gensim
import nltk

print(nltk.corpus.movie_reviews.fileids())
# ['neg/cv000_29416.txt', 'neg/cv001_19502.txt', ...]

# print(nltk.corpus.movie_reviews.raw('neg/cv000_29416.txt'))     # 문자열, str
# print(nltk.corpus.movie_reviews.words('neg/cv000_29416.txt'))   # 1차원 리스트, []
# print(nltk.corpus.movie_reviews.sents('neg/cv000_29416.txt'))   # 2차원 리스트, [[]]

sents = nltk.corpus.movie_reviews.sents()
model = gensim.models.Word2Vec(sents)

print(model)
# Word2Vec(vocab=14794, size=100, alpha=0.025)

# 코사인 유사도 : 같은 방향(1), 반대 방향(-1), 직교(0)
print(model.wv.similarity('villain', 'hero'))   # 0.6182198
print(model.wv.similarity('man', 'woman'))      # 0.89994997
print(model.wv.similarity('sky', 'earth'))      # 0.5745624
print(model.wv.similarity('smile', 'space'))    # 0.49382892

print(model.wv.most_similar('villain'))
# [('impression', 0.8033424019813538), ('convincing', 0.7754334807395935),
# ('charismatic', 0.7650010585784912), ('annoying', 0.756291925907135),
# ('actress', 0.7562599778175354), ('actor', 0.7550784349441528),
# ('poor', 0.7423707842826843), ('heroine', 0.7423393726348877),
# ('result', 0.7418272495269775), ('casting', 0.739983081817627)]

# print('villain' in model.wv)
