# Day_03_01_Gensim.py
# http://210.125.150.125/
import gensim   #topic modeling
from pprint import pprint
import nltk
import collections

def show_doc2vec():
    f = open('data/computer.txt', 'r', encoding='utf-8')
    documents = [line.strip() for line in f]#개행문자 삭제
    f.close()

    print(documents)

    #vector화하기 위한 작업

    # 문제
    # documents에 포함된 항목을 개별적요소 단어들로 변환하세요.
    # [['Human', 'machine', 'interface', 'for', 'lab', 'abc',...]]
    # documents는 라인이

    doc_list = [doc.lower().split() for doc in documents]#문장이 하나씩 불러온다. 분량이 많지 않으므로 tokenizer를 사용하지 않았다.
    pprint(doc_list)#개행문자등을 구분하기 좋도록 출력해주는 기능이 있다.

    print('-'*50)

    #문제
    # 커스텀 stop-words를 만들어서 불필요한 단어를 제거하세요.

    stop_words = ['in', 'for', 'abs', 'of', 'and','a']
    # doc_list = [[w for w in words ] for words in doc_list]#원본과 같은 내용
    # if a not in stop_words


    doc_list = [[w for w in words if w not in stop_words] for words in doc_list]#
    pprint(doc_list)

    print('-'*50)
    # 문제
    # 출현 빈도가 1인 단어를 삭제하세요.
    # 출현 빈도수
    # freq = []
    # for w in doc_list:
    #     freq[w] = freq.get(w, 0) + 1
    # print(freq)
    # doc_list2 = [w for w in doc_list if len(w)>1]
    # pprint(doc_list2)

    # 출현 빈도수
    freq = nltk.FreqDist(w for words in doc_list for w in words)
    print(freq.most_common())
    # [('system', 4), ('user', 3), ('the', 3), ('trees', 3), ('graph', 3)..]

    # 출현 빈도수
    # freq = nltk.FreqDist(w for words in doc_list for w in words)
    # print(freq.most_common())
    # [('system', 4), ('user', 3), ('the', 3), ('trees', 3),...]
    print(11)

    # 출현 빈도수 1인 단어 삭제
    doc_list = [[w for w in words if freq[w] > 1] for words in doc_list]
    pprint(doc_list)
    # [['human', 'interface', 'computer'],..]
    print(22)

    dct = gensim.corpora.Dictionary(doc_list)
    print(dct)
    # Dictionary(14 unique tokens: ['computer', 'human', 'interface', 'a',

    print(dct.token2id)
    # {'computer': 0, 'human': 1, 'interface': 2, 'a': 3, 'response': 4,...}

    # bow : bog of wrds
    print(dct.doc2bow(['computer', 'trees', 'graph', 'trees']))
    # [(0, 1), (11, 2), (12, 1)]


    # 문제
    # 문서 전체 (doc_list)를 벡터로 변환하세요.
    # print(dct.doc2bow(doc_list))
    vectors =[dct.doc2bow(words) for words in doc_list]
    print(vectors)

# word to bag rnn의 핵심 endcoding 방법이다.

def show_word2vec():
    def save():
        text  = ['나는 너를 사랑해', '나는 너를 미워해']
        token = [s.split() for s in text]
        # print(token)
        # 나는 : 0, (1,0), (0,1), 0.2,[(0.1, 0.5)] 이 값은 좌표처럼 활용이 가능하지 않나.단어의 거리 similarty의 측정이 가능하다.
        # 너를 : 1, (0,1), (1,0), 0.4,(0.7, 0.2)

        # embedding = gensim.models.Word2Vec(token, min_count=1, size=5)
        embedding = gensim.models.Word2Vec(token, min_count=1, size=5, sg=True)# 정확도가 잘 나오지만 서너배 정도 데이터가 많아진다.

        print(embedding)
        # Word2Vec(vocab=4, size=5, alpha=0.025)

        print(embedding.wv.vectors)
        # [[ 0.07327038 -0.09387692 -0.06886869 -0.06110539  0.06266866]...]의미있는 숫자로 출력이 된다.
        print(embedding.wv.vectors.shape)
        # (4, 5)

        print(embedding.wv['나는'])
        # [ 0.08054517 -0.08366419  0.08401859 -0.02942456 -0.04241462]

        embedding.save('data/word2vec.out')

    embedding =gensim.models.Word2Vec.load('data/word2vec.out')
    print(embedding.wv['나는'])
    save()
# show_doc2vec()
show_word2vec()