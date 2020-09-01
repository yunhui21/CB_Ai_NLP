# Day_04_01_gensim.py
import gensim           # topic modeling
from pprint import pprint
import nltk
import collections


def show_doc2vec():
    f = open('data/computer.txt', 'r', encoding='utf-8')
    documents = [line.strip() for line in f]
    f.close()

    print(documents)

    # 문제
    # documents에 포함된 항목을 개별적으로 단어들로 변환하세요
    # [['Human', 'machine', 'interface'], [...]]
    doc_list = [doc.lower().split() for doc in documents]
    pprint(doc_list)

    # 문제
    # 커스텀 stop-words를 만들어서 불필요한 단어를 제거하세요
    stop_words = ['of', 'and', 'the', 'to', 'a', 'for', 'in']
    doc_list = [[w for w in words if w not in stop_words] for words in doc_list]
    pprint(doc_list)

    # 문제
    # 출현 빈도가 1인 단어를 삭제하세요
    # freq = nltk.FreqDist(w for words in doc_list for w in words)
    freq = collections.Counter(w for words in doc_list for w in words)
    # print(freq.most_common())

    doc_list = [[w for w in words if freq[w] > 1] for words in doc_list]
    pprint(doc_list)

    dct = gensim.corpora.Dictionary(doc_list)
    print(dct)
    # Dictionary(12 unique tokens: ['computer', 'human', 'interface', 'response', 'survey']...)

    print(dct.token2id)
    # {'computer': 0, 'human': 1, 'interface': 2, 'response': 3, 'survey': 4, 'system': 5,
    # 'time': 6, 'user': 7, 'eps': 8, 'trees': 9, 'graph': 10, 'minors': 11}

    # bow : bag of words
    print(dct.doc2bow(['computer', 'trees', 'graph', 'trees']))
    # [(0, 1), (9, 2), (10, 1)]

    # 문제
    # 문서 전체(doc_list)를 벡터로 변환하세요
    vectors = [dct.doc2bow(words) for words in doc_list]
    pprint(vectors)


def show_word2vec():
    def save():
        text = ['나는 너를 사랑해', '나는 너를 미워해']
        token = [s.split() for s in text]
        # print(token)
        # [['나는', '너를', '사랑해'], ['나는', '너를', '미워해']]
        # 나는 : 0, (1, 0), 0.2, (0.1, -0.5)
        # 너를 : 1, (0, 1), 0.4, (0.7, 0.2)

        # embedding = gensim.models.Word2Vec(token, min_count=1, size=5)
        embedding = gensim.models.Word2Vec(token, min_count=1, size=5, sg=True)
        print(embedding)
        # Word2Vec(vocab=4, size=5, alpha=0.025)

        print(embedding.wv.vectors)
        print(embedding.wv.vectors.shape)       # (4, 5)

        print(embedding.wv['나는'])

        embedding.save('data/word2vec.out')

    save()

    # embedding = gensim.models.Word2Vec.load('data/word2vec.out')
    # print(embedding.wv['나는'])


# show_doc2vec()
show_word2vec()
