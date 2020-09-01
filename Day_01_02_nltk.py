import nltk
import string

# nltk.download('gutenberg')
# nltk.download('stopwords')
# nltk.download('webtest')
# nltk.download('wordnet')
# nltk.download('reuters')
# nltk.download('averaged_percenptron_tagger')
# nltk.download('punkt')

def dataset():
    nltk.download('gutenberg')
    nltk.download('stopwords')
    nltk.download('webtest')
    nltk.download('wordnet')
    nltk.download('reuters')
    nltk.download('averaged_percenptron_tagger')
    nltk.download('punkt')

def corpus():

    print(nltk.corpus.gutenberg)

    print(nltk.corpus.gutenberg.fileids())

    print(nltk.corpus.gutenberg)

    emma = nltk.corpus.gutenberg.raw('austen-emma.txt')
    print(emma)
    print(type(emma))

    print(nltk.corpus.gutenberg.words())

def tokenize():
    text = nltk.corpus.gutenberg.raw('austen-emma.txt')
    text = text[:1000]
    print(text)

    print(nltk.tokenize.simple.SpaceTokenizer().tokenize(text))
    print(nltk.tokenize.sent_tokenize(text))

    for sent in nltk.tokenize.sent_tokenize(text):
        print(sent)
        print('------------------')

    print(nltk.tokenize.RegexpTokenizer(r'[a-zA-Z]+').tokenize(text))
    print(nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(text))

    print(nltk.tokenize.WordPuntTokenizer().tokenize(text))

#어간추출

def setemming():
    words = ['lives', 'dies', 'files', 'died']

    st = nltk.stem.PorterStemmer()
    print(st.stem('lives'))
    print([st.stem(w) for w in words])

    st = nltk.stem.LancasterStemmer()
    print([st.stem(w) for w in words])

def grams():
    text = 'Jonh works at Intel'
    tokens = nltk.word_tokenize(text)

    print(tokens)
    print(list(nltk.bigrams(tokens)))
    print(list(nltk.tigrams(tokens)))