# Day_01_02_nltk.py
import nltk         # natural language tool kit
import string


def datasets():
    nltk.download('gutenberg')
    nltk.download('stopwords')
    nltk.download('webtext')
    nltk.download('wordnet')
    nltk.download('reuters')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt')

    # nltk.download()


def corpus():
    print(nltk.corpus.gutenberg)

    print(nltk.corpus.gutenberg.fileids())
    # ['austen-emma.txt', 'austen-persuasion.txt', ...]

    print(nltk.corpus.gutenberg)

    emma = nltk.corpus.gutenberg.raw('austen-emma.txt')
    print(emma)
    print(type(emma))       # <class 'str'>

    print(nltk.corpus.gutenberg.words())


def tokenize():
    text = nltk.corpus.gutenberg.raw('austen-emma.txt')
    text = text[:1000]
    print(text)

    print(nltk.tokenize.simple.SpaceTokenizer().tokenize(text))
    print(nltk.tokenize.sent_tokenize(text))

    for sent in nltk.tokenize.sent_tokenize(text):
        print(sent)
        print('----------')

    print(nltk.tokenize.RegexpTokenizer(r'[a-zA-Z]+').tokenize(text))
    print(nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(text))

    print(nltk.tokenize.WordPunctTokenizer().tokenize(text))


# 어간 추출
def stemming():
    words = ['lives', 'dies', 'flies', 'died']

    st = nltk.stem.PorterStemmer()
    print(st.stem('lives'))
    print([st.stem(w) for w in words])

    st = nltk.stem.LancasterStemmer()
    print([st.stem(w) for w in words])


def grams():
    text = 'John works at Intel'
    tokens = nltk.word_tokenize(text)

    print(tokens)
    print(list(nltk.bigrams(tokens)))
    print(list(nltk.trigrams(tokens)))


# datasets()
# corpus()
# tokenize()
# stemming()
grams()

# applekoong@naver.com 김정훈





